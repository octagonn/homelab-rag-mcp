#!/usr/bin/env python3
"""Markdown vault indexer with SQLite FTS5, embeddings, and wikilinks.
Public edition defaults to fictional sample documents only.
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / ".data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "index.db"
MANIFEST_PATH = DATA_DIR / "MANIFEST.md"
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384

WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
MAX_CHUNK_CHARS = 1400


# --------------------------------------------------------------------------- #
# Vault resolution
# --------------------------------------------------------------------------- #
def resolve_vault() -> Path:
    configured = os.environ.get("SERVER_BRAIN_VAULT")
    vault = Path(configured).expanduser().resolve() if configured else HERE / "sample_vault"
    if not vault.is_dir():
        raise ValueError("Configured document directory does not exist")
    return vault



def iter_md_files(vault: Path):
    for p in vault.rglob("*.md"):
        if p.is_symlink() or not p.resolve().is_relative_to(vault.resolve()) or ".obsidian" in p.parts or ".trash" in p.parts:
            continue
        yield p


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_frontmatter(text: str):
    """Return (frontmatter_dict, body). Frontmatter is optional."""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            raw = text[3:end].strip("\n")
            body = text[end + 4 :].lstrip("\n")
            try:
                fm = yaml.safe_load(raw) or {}
                if isinstance(fm, dict):
                    return fm, body
            except yaml.YAMLError:
                pass
    return {}, text


def as_list(v):
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    return [str(v)]


def first_heading(body: str):
    for line in body.splitlines():
        m = HEADING_RE.match(line)
        if m:
            return m.group(2).strip()
    return None


def normalize_link_target(raw: str) -> str:
    """'[[Nodes/demo-node|Demo]]' -> 'demo-node' (note stem)."""
    raw = raw.replace("\\|", "|")          # table-escaped pipe
    target = raw.split("|", 1)[0].strip()  # drop display alias
    target = target.split("#", 1)[0].strip()  # drop heading anchor
    target = target.rsplit("/", 1)[-1]     # drop path
    if target.lower().endswith(".md"):
        target = target[:-3]
    return target.strip()


def extract_links(body: str):
    out = []
    for m in WIKILINK_RE.finditer(body):
        raw = m.group(1)
        norm = normalize_link_target(raw)
        if norm:
            out.append((raw.replace("\\|", "|"), norm))
    return out


def strip_md_noise(text: str) -> str:
    """Light cleanup for embedding/snippet text: callout markers, table pipes,
    standalone horizontal rules."""
    text = re.sub(r"^>\s?\[![a-z]+\][-+]?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*([-*=_])\1{2,}\s*$", "", text, flags=re.MULTILINE)
    text = text.replace("\\|", "|")
    return text.strip()


def chunk_body(title: str, body: str):
    """Split into (heading, text) sections; sub-split overlong sections."""
    sections = []
    cur_heading = title
    cur_lines = []

    def flush():
        text = "\n".join(cur_lines).strip()
        if text:
            sections.append((cur_heading, text))

    for line in body.splitlines():
        m = HEADING_RE.match(line)
        if m:
            flush()
            cur_heading = m.group(2).strip()
            cur_lines = []
        else:
            cur_lines.append(line)
    flush()

    chunks = []
    for heading, text in sections:
        text = strip_md_noise(text)
        if len(text) <= MAX_CHUNK_CHARS:
            chunks.append((heading, text))
            continue
        # window on paragraph boundaries
        buf = ""
        for para in re.split(r"\n\s*\n", text):
            if len(buf) + len(para) + 2 > MAX_CHUNK_CHARS and buf:
                chunks.append((heading, buf.strip()))
                buf = ""
            buf += para + "\n\n"
        if buf.strip():
            chunks.append((heading, buf.strip()))
    return chunks


# --------------------------------------------------------------------------- #
# DB
# --------------------------------------------------------------------------- #
def open_db():
    db = sqlite3.connect(DB_PATH)
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE IF NOT EXISTS notes(
            id INTEGER PRIMARY KEY,
            rel_path TEXT UNIQUE, abs_path TEXT,
            title TEXT, type TEXT, node TEXT,
            tags TEXT, aliases TEXT, updated TEXT,
            mtime REAL, content_hash TEXT, body TEXT
        );
        CREATE TABLE IF NOT EXISTS chunks(
            id INTEGER PRIMARY KEY,
            note_id INTEGER, heading TEXT, text TEXT, embedding BLOB
        );
        CREATE TABLE IF NOT EXISTS links(
            src_note_id INTEGER, target_raw TEXT, target_norm TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_note ON chunks(note_id);
        CREATE INDEX IF NOT EXISTS idx_links_src ON links(src_note_id);
        CREATE INDEX IF NOT EXISTS idx_links_norm ON links(target_norm);
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text, title, tags, headings
        );
        """
    )
    return db


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def delete_note(db, note_id):
    cur = db.execute("SELECT id FROM chunks WHERE note_id=?", (note_id,))
    for (cid,) in cur.fetchall():
        db.execute("DELETE FROM chunks_fts WHERE rowid=?", (cid,))
    db.execute("DELETE FROM chunks WHERE note_id=?", (note_id,))
    db.execute("DELETE FROM links WHERE src_note_id=?", (note_id,))
    db.execute("DELETE FROM notes WHERE id=?", (note_id,))


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build(rebuild: bool = False):
    import numpy as np
    from sentence_transformers import SentenceTransformer

    vault = resolve_vault()
    db = open_db()

    if rebuild:
        db.executescript(
            "DELETE FROM notes; DELETE FROM chunks; DELETE FROM links; "
            "DELETE FROM chunks_fts;"
        )
        db.commit()

    existing = {r[0]: (r[1], r[2]) for r in db.execute(
        "SELECT rel_path, id, content_hash FROM notes"
    )}
    seen = set()
    pending = []  # (note_id, [(heading, embed_text, raw_text)])

    print(f"Vault: {vault}", file=sys.stderr)
    for path in iter_md_files(vault):
        rel = str(path.relative_to(vault)).replace("\\", "/")
        seen.add(rel)
        data = path.read_bytes()
        h = file_hash(data)
        mtime = path.stat().st_mtime

        if rel in existing and existing[rel][1] == h:
            continue  # unchanged

        if rel in existing:
            delete_note(db, existing[rel][0])

        text = data.decode("utf-8", errors="replace")
        fm, body = parse_frontmatter(text)
        title = (fm.get("title") or first_heading(body) or path.stem).strip()
        tags = as_list(fm.get("tags"))
        aliases = as_list(fm.get("aliases"))
        node = fm.get("node") or ""
        ntype = fm.get("type") or ""
        updated = str(fm.get("updated") or "")

        cur = db.execute(
            "INSERT INTO notes(rel_path,abs_path,title,type,node,tags,aliases,"
            "updated,mtime,content_hash,body) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (rel, str(path), title, ntype, node, json.dumps(tags),
             json.dumps(aliases), updated, mtime, h, body),
        )
        note_id = cur.lastrowid

        for raw, norm in extract_links(body):
            db.execute(
                "INSERT INTO links(src_note_id,target_raw,target_norm) VALUES(?,?,?)",
                (note_id, raw, norm),
            )

        chunk_rows = []
        for heading, ctext in chunk_body(title, body):
            embed_text = f"{title} — {heading}\n{ctext}"
            chunk_rows.append((heading, embed_text, ctext))
        pending.append((note_id, tags, title, chunk_rows))

    # Remove notes whose files are gone
    for rel, (nid, _h) in existing.items():
        if rel not in seen:
            delete_note(db, nid)
            print(f"  removed deleted note: {rel}", file=sys.stderr)

    # Embed all pending chunks in one batch
    total_chunks = sum(len(r[3]) for r in pending)
    if total_chunks:
        print(f"Embedding {total_chunks} new/changed chunks "
              f"across {len(pending)} notes...", file=sys.stderr)
        model = SentenceTransformer(EMBED_MODEL)
        flat_texts = [c[1] for _, _, _, rows in pending for c in rows]
        vecs = model.encode(
            flat_texts, batch_size=64, normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        i = 0
        for note_id, tags, title, rows in pending:
            tagstr = " ".join(tags)
            for heading, _embed_text, raw_text in rows:
                emb = vecs[i].tobytes()
                i += 1
                cur = db.execute(
                    "INSERT INTO chunks(note_id,heading,text,embedding) "
                    "VALUES(?,?,?,?)",
                    (note_id, heading, raw_text, emb),
                )
                cid = cur.lastrowid
                db.execute(
                    "INSERT INTO chunks_fts(rowid,text,title,tags,headings) "
                    "VALUES(?,?,?,?,?)",
                    (cid, raw_text, title, tagstr, heading),
                )
    else:
        print("No changes to embed.", file=sys.stderr)

    db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('embed_model',?)",
               (EMBED_MODEL,))
    db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('embed_dim',?)",
               (str(EMBED_DIM),))
    db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('built_at',?)",
               (str(time.time()),))
    db.commit()

    n_notes = db.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    n_chunks = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    db.close()
    print(f"Index ready: {n_notes} notes, {n_chunks} chunks -> {DB_PATH}",
          file=sys.stderr)


# --------------------------------------------------------------------------- #
# Manifest (no model required)
# --------------------------------------------------------------------------- #
def extract_home_table(home_text: str, section_marker: str):
    """Pull the first markdown table that appears after a section marker."""
    idx = home_text.find(section_marker)
    if idx == -1:
        return []
    rows = []
    started = False
    for line in home_text[idx:].splitlines():
        s = line.strip()
        if s.startswith("|"):
            started = True
            protected = s.strip("|").replace("\\|", "\x00")  # guard escaped pipes
            cells = [c.replace("\x00", "|").strip() for c in protected.split("|")]
            if set("".join(cells)) <= set("-: "):  # separator row
                continue
            rows.append(cells)
        elif started:
            break  # markdown tables are contiguous; first non-row line ends it
    return rows


def generate_manifest():
    vault = resolve_vault()
    home = (vault / "HOME.md")
    home_text = home.read_text("utf-8", errors="replace") if home.exists() else ""

    # Build doc map grouped by top-level folder
    notes = []
    for path in iter_md_files(vault):
        rel = str(path.relative_to(vault)).replace("\\", "/")
        text = path.read_text("utf-8", errors="replace")
        fm, body = parse_frontmatter(text)
        title = (fm.get("title") or first_heading(body) or path.stem).strip()
        title = re.sub(r"[^\x00-\x7F]+", "", title).strip().strip("—- ")
        notes.append((rel, title))
    notes.sort()

    groups = {}
    for rel, title in notes:
        top = rel.split("/", 1)[0] if "/" in rel else "(root)"
        groups.setdefault(top, []).append((rel, title))

    out = []
    out.append("# Server-Brain Manifest (auto-generated)\n")
    out.append(
        "> Compact index of the **Server Info** vault "
        f"(`{vault}`). For full content use the **server-brain** MCP tools: "
        "`vault_search`, `get_note`, `get_service`, `get_node`, `whereis`, "
        "`backlinks`. Regenerated each session from the vault; do not edit by hand.\n"
    )

    nodes = extract_home_table(home_text, "Nodes at a Glance")
    if nodes:
        out.append("## Nodes")
        out.append("| " + " | ".join(nodes[0]) + " |")
        out.append("|" + "|".join(["---"] * len(nodes[0])) + "|")
        for r in nodes[1:]:
            r = [re.sub(r"\[\[[^\]]*?\|?([^\]|]+)\]\]", r"\1", c) for c in r]
            r = [c.replace("**", "").replace("🟦", "").replace("🟩", "")
                  .replace("🟧", "").replace("🟥", "").replace("🟨", "")
                  .replace("🟪", "").replace("⬜", "").strip() for c in r]
            out.append("| " + " | ".join(r) + " |")
        out.append("")

    services = extract_home_table(home_text, "Service Quick-Links")
    if services:
        out.append("## Services")
        out.append("| " + " | ".join(services[0]) + " |")
        out.append("|" + "|".join(["---"] * len(services[0])) + "|")
        for r in services[1:]:
            out.append("| " + " | ".join(r) + " |")
        out.append("")

    out.append("## Doc Map (every note -> open with get_note)")
    for top in sorted(groups):
        out.append(f"\n**{top}/**")
        for rel, title in groups[top]:
            out.append(f"- `{rel}` - {title}")
    out.append("")

    MANIFEST_PATH.write_text("\n".join(out), encoding="utf-8")
    print(f"Manifest written: {MANIFEST_PATH} ({len(notes)} notes)",
          file=sys.stderr)


# --------------------------------------------------------------------------- #
# Staleness
# --------------------------------------------------------------------------- #
def check_stale() -> bool:
    vault = resolve_vault()
    if not DB_PATH.exists():
        return True
    db_mtime = DB_PATH.stat().st_mtime
    files = list(iter_md_files(vault))
    for path in files:
        if path.stat().st_mtime > db_mtime:
            return True
    # detect added/removed notes even when no surviving file is newer
    try:
        db = sqlite3.connect(DB_PATH)
        n = db.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        db.close()
    except sqlite3.Error:
        return True
    return n != len(files)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--manifest-only", action="store_true")
    ap.add_argument("--check-stale", action="store_true")
    args = ap.parse_args()

    if args.check_stale:
        sys.exit(1 if check_stale() else 0)
    if args.manifest_only:
        generate_manifest()
        return
    build(rebuild=args.rebuild)
    generate_manifest()


if __name__ == "__main__":
    main()
