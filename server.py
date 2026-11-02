#!/usr/bin/env python3
"""MCP retrieval tools over a configured Markdown vault.
Public edition uses fictional sample documents by default.
"""

import json
import logging
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import brain_index as bi  # noqa: E402

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
log = logging.getLogger("server-brain")

DB_PATH = bi.DB_PATH
RRF_K = 60

mcp = FastMCP("server-brain")

# --------------------------------------------------------------------------- #
# Startup: refresh-if-stale, then load model + embedding matrix into memory
# --------------------------------------------------------------------------- #
log.info("Loading embedding model %s ...", bi.EMBED_MODEL)
_model = SentenceTransformer(bi.EMBED_MODEL)

if not DB_PATH.exists() or bi.check_stale():
    log.info("Index missing or stale — rebuilding (incremental)...")
    try:
        bi.build(rebuild=False)
        bi.generate_manifest()
    except Exception as e:  # noqa: BLE001
        log.error("Rebuild failed: %s", e)


class Index:
    """In-memory view of the chunk embeddings + note metadata."""

    def __init__(self):
        self.matrix = None        # (n_chunks, dim) float32, L2-normalized
        self.chunk_meta = []      # parallel: dict per chunk
        self.load()

    def load(self):
        db = sqlite3.connect(DB_PATH)
        rows = db.execute(
            "SELECT c.id, c.note_id, c.heading, c.text, c.embedding, "
            "n.rel_path, n.title FROM chunks c JOIN notes n ON n.id=c.note_id"
        ).fetchall()
        db.close()
        vecs, meta = [], []
        for cid, note_id, heading, text, emb, rel, title in rows:
            vecs.append(np.frombuffer(emb, dtype=np.float32))
            meta.append({"chunk_id": cid, "note_id": note_id, "heading": heading,
                         "text": text, "rel_path": rel, "title": title})
        self.matrix = (np.vstack(vecs) if vecs
                       else np.zeros((0, bi.EMBED_DIM), dtype=np.float32))
        self.chunk_meta = meta
        log.info("Loaded %d chunks into memory.", len(meta))


_idx = Index()


# --------------------------------------------------------------------------- #
# Search helpers
# --------------------------------------------------------------------------- #
def _fts_query(query: str) -> str:
    toks = re.findall(r"[A-Za-z0-9_.]{2,}", query)
    toks = [t for t in toks if t.upper() not in ("AND", "OR", "NOT", "NEAR")]
    return " OR ".join(f'"{t}"' for t in toks)


def _semantic(query: str, n: int):
    if _idx.matrix.shape[0] == 0:
        return []
    q = _model.encode([query], normalize_embeddings=True,
                      show_progress_bar=False)[0].astype("float32")
    scores = _idx.matrix @ q
    order = np.argsort(-scores)[:n]
    return [(int(i), float(scores[i])) for i in order]


def _keyword(query: str, n: int):
    fq = _fts_query(query)
    if not fq:
        return []
    db = sqlite3.connect(DB_PATH)
    try:
        # Columns: (text, title, tags, headings). Weight title/headings/tags
        # higher so a note whose title or heading matches the query outranks
        # an incidental body mention.
        rows = db.execute(
            "SELECT rowid, bm25(chunks_fts, 1.0, 3.0, 2.0, 2.0) AS r "
            "FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY r LIMIT ?",
            (fq, n),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        db.close()
    by_id = {m["chunk_id"]: i for i, m in enumerate(_idx.chunk_meta)}
    return [(by_id[r[0]], r[1]) for r in rows if r[0] in by_id]


def _hybrid(query: str, limit: int):
    sem = _semantic(query, 25)
    kw = _keyword(query, 25)
    fused = {}
    for rank, (i, _s) in enumerate(sem):
        fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank)
    for rank, (i, _s) in enumerate(kw):
        fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank)
    ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:limit]
    out = []
    for i, score in ranked:
        m = _idx.chunk_meta[i]
        snippet = " ".join(m["text"].split())[:320]
        out.append({"rel_path": m["rel_path"], "title": m["title"],
                    "heading": m["heading"], "snippet": snippet,
                    "score": round(score, 4)})
    return out


def _conn():
    return sqlite3.connect(DB_PATH)


def _resolve_candidates(name: str):
    """Return (exact, loose) candidate note dicts for a name/path/title/alias.
    'exact' = rel_path / stem / title / alias matches (rel_path ranked first);
    'loose' = substring matches on title or path."""
    db = _conn()
    db.row_factory = sqlite3.Row
    rows = db.execute("SELECT * FROM notes").fetchall()
    db.close()
    name_l = name.strip().lower()
    stem = name_l.rsplit("/", 1)[-1]
    if stem.endswith(".md"):
        stem = stem[:-3]
    exact, loose = [], []
    for r in rows:
        rp = r["rel_path"].lower()
        rp_stem = rp.rsplit("/", 1)[-1]
        if rp_stem.endswith(".md"):
            rp_stem = rp_stem[:-3]
        try:
            aliases = [a.lower() for a in json.loads(r["aliases"])]
        except Exception:  # noqa: BLE001
            aliases = []
        if rp == name_l or rp == name_l + ".md":
            exact.insert(0, dict(r))                 # full rel_path is strongest
        elif rp_stem == stem or r["title"].lower() == name_l or name_l in aliases:
            exact.append(dict(r))
        elif name_l and (name_l in r["title"].lower() or stem in rp):
            loose.append(dict(r))
    return exact, loose


def _resolve_note(name: str):
    """Best single match (for backlinks/get_service). Returns dict|None."""
    exact, loose = _resolve_candidates(name)
    cands = exact or loose
    return cands[0] if cands else None


def _backlinks_for(note_id: int, rel_path: str):
    stem = rel_path.rsplit("/", 1)[-1][:-3]
    db = _conn()
    db.row_factory = sqlite3.Row
    srcs = db.execute(
        "SELECT DISTINCT n.rel_path, n.title FROM links l "
        "JOIN notes n ON n.id=l.src_note_id WHERE l.target_norm=? COLLATE NOCASE",
        (stem,),
    ).fetchall()
    db.close()
    return [{"rel_path": r["rel_path"], "title": r["title"]} for r in srcs]


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
@mcp.tool()
def vault_search(query: str, limit: int = 6) -> str:
    """Hybrid semantic + keyword search over the entire Server Info vault.
    Use this for any question about demo-node / the homelab when you don't
    already know the exact note. Returns ranked matches with file path,
    heading, and a snippet. Follow up with get_note for full content."""
    if not query or not query.strip():
        return "Provide a search query (e.g. a service, host, port, or topic)."
    results = _hybrid(query, max(1, min(limit, 20)))
    if not results:
        return f"No matches for: {query!r}"
    lines = [f"Top {len(results)} matches for {query!r}:\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. [{r['rel_path']}] — {r['title']} › {r['heading']}")
        lines.append(f"   {r['snippet']}")
    lines.append("\n(Use get_note('<rel_path>') for the full note.)")
    return "\n".join(lines)


@mcp.tool()
def get_note(name_or_path: str) -> str:
    """Return the full content of one note plus its outgoing links and
    backlinks. Accepts a rel_path (e.g. 'Nodes/demo-node.md'),
    a note stem, title, or alias. If several notes match, lists them."""
    exact, loose = _resolve_candidates(name_or_path)
    if len(exact) > 1:
        return ("Multiple notes match exactly — call get_note with one of:\n" +
                "\n".join(f"  - {n['rel_path']} — {n['title']}" for n in exact))
    note = exact[0] if exact else (loose[0] if loose else None)
    if not note:
        return (f"No note matches {name_or_path!r}. "
                f"Try vault_search to find it.")

    # Prefer the live file so content is never stale vs a mid-session edit.
    title, body = note["title"], note["body"]
    ntype, node, updated = note["type"], note["node"], note["updated"]
    tags_list = json.loads(note["tags"])
    try:
        p = Path(note["abs_path"]).resolve()
        if not p.is_relative_to(bi.resolve_vault().resolve()):
            raise ValueError("Note outside configured vault")
        if p.exists():
            fm, b = bi.parse_frontmatter(p.read_text("utf-8", errors="replace"))
            body = b
            title = fm.get("title") or title
            tags_list = bi.as_list(fm.get("tags")) or tags_list
            ntype = fm.get("type") or ntype
            node = fm.get("node") or node
            updated = str(fm.get("updated") or updated)
    except Exception:  # noqa: BLE001
        pass

    db = _conn()
    db.row_factory = sqlite3.Row
    out_links = db.execute(
        "SELECT DISTINCT target_norm FROM links WHERE src_note_id=?",
        (note["id"],),
    ).fetchall()
    db.close()
    back = _backlinks_for(note["id"], note["rel_path"])
    header = (f"# {title}\n"
              f"path: {note['rel_path']} | type: {ntype or '—'} | "
              f"node: {node or '—'} | updated: {updated or '—'}\n"
              f"tags: {', '.join(tags_list) or '—'}\n")
    links = ", ".join(sorted({r["target_norm"] for r in out_links})) or "—"
    bl = ", ".join(b["rel_path"] for b in back) or "—"
    out = f"{header}\n{body}\n\n---\nLinks to: {links}\nBacklinks: {bl}"
    others = [n["rel_path"] for n in loose if n["rel_path"] != note["rel_path"]]
    if not exact and others:
        out += "\nOther notes matching: " + ", ".join(others[:6])
    return out


@mcp.tool()
def get_service(name: str) -> str:
    """Everything documented about a service (e.g. 'jellyfin', 'amp',
    'qbittorrent', 'adguard'): host, port, URL, config/compose paths, and the
    relevant guide. Combines tag matches with a hybrid search."""
    tag = f"service/{name.strip().lower()}"
    db = _conn()
    db.row_factory = sqlite3.Row
    rows = db.execute("SELECT rel_path, title, tags FROM notes").fetchall()
    db.close()
    hits = [r for r in rows if tag in [t.lower() for t in json.loads(r["tags"])]]
    parts = [f"## Service: {name}"]
    if hits:
        parts.append("Tagged notes: " +
                     ", ".join(f"{h['rel_path']}" for h in hits))
    parts.append("\n" + vault_search(f"{name} service port url config path", 6))
    return "\n".join(parts)


@mcp.tool()
def get_node(name: str) -> str:
    """Everything documented about a node/host (e.g. 'demo-node',
    'demo-worker', 'demo-lab', 'demo-monitor'): guide, hardware spec, and
    configuration notes, plus a hybrid search."""
    nm = name.strip().lower()
    tag = f"node/{nm}"
    db = _conn()
    db.row_factory = sqlite3.Row
    rows = db.execute("SELECT rel_path, title, tags, node FROM notes").fetchall()
    db.close()
    hits = [r for r in rows
            if tag in [t.lower() for t in json.loads(r["tags"])]
            or (r["node"] or "").lower() == nm]
    parts = [f"## Node: {name}"]
    if hits:
        parts.append("Related notes:")
        for h in hits:
            parts.append(f"  - {h['rel_path']} — {h['title']}")
    parts.append("\n" + vault_search(f"{name} node host ip role services", 6))
    return "\n".join(parts)


@mcp.tool()
def whereis(query: str) -> str:
    """Answer 'which file / where is X documented or stored?' — returns the
    most relevant note paths with snippets. Best for locating a config path,
    configuration, or runbook quickly."""
    if not query or not query.strip():
        return "Provide something to locate (a path, service, configuration, or topic)."
    results = _hybrid(query, 8)
    if not results:
        return f"No matches for: {query!r}"
    lines = [f"Most likely locations for {query!r}:"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['rel_path']} › {r['heading']}  —  {r['snippet'][:220]}")
    return "\n".join(lines)


@mcp.tool()
def backlinks(name_or_path: str) -> str:
    """List notes that link TO the given note (incoming wikilinks)."""
    note = _resolve_note(name_or_path)
    if not note:
        return f"No note matches {name_or_path!r}."
    back = _backlinks_for(note["id"], note["rel_path"])
    if not back:
        return f"No backlinks to {note['rel_path']}."
    return f"Backlinks to {note['rel_path']}:\n" + "\n".join(
        f"  - {b['rel_path']} — {b['title']}" for b in back)


@mcp.tool()
def list_services() -> str:
    """List all documented services (from #service/* tags)."""
    db = _conn()
    rows = db.execute("SELECT tags FROM notes").fetchall()
    db.close()
    svcs = set()
    for (tj,) in rows:
        for t in json.loads(tj):
            if t.lower().startswith("service/"):
                svcs.add(t.split("/", 1)[1])
    return "Documented services: " + ", ".join(sorted(svcs)) if svcs else \
        "No #service/* tags found; try vault_search."


@mcp.tool()
def list_nodes() -> str:
    """List all documented nodes/hosts (from #node/* tags)."""
    db = _conn()
    rows = db.execute("SELECT tags FROM notes").fetchall()
    db.close()
    nodes = set()
    for (tj,) in rows:
        for t in json.loads(tj):
            if t.lower().startswith("node/"):
                nodes.add(t.split("/", 1)[1])
    return "Documented nodes: " + ", ".join(sorted(nodes))


@mcp.tool()
def get_manifest() -> str:
    """Return the compact always-on manifest (nodes, services, doc map)."""
    if bi.MANIFEST_PATH.exists():
        return bi.MANIFEST_PATH.read_text("utf-8", errors="replace")
    return "Manifest not generated yet; run reindex."


@mcp.tool()
def reindex() -> str:
    """Refresh the index and manifest from the vault (incremental). Use after
    editing vault notes so search reflects the changes without a restart."""
    bi.build(rebuild=False)
    bi.generate_manifest()
    _idx.load()
    return (f"Reindexed. {len(_idx.chunk_meta)} chunks loaded; "
            f"manifest regenerated.")


if __name__ == "__main__":
    mcp.run()
