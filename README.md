# Homelab RAG MCP

An MCP server for retrieving Markdown infrastructure documentation using local
embeddings, SQLite FTS5, reciprocal rank fusion, and an Obsidian wikilink graph.
This is the existing Server-Brain implementation with private paths removed;
only the included fictional vault is indexed by default.

## Run

Use Python 3.11 or newer in a virtual environment.

```sh
python -m pip install -r requirements.txt
python brain_index.py --rebuild
python server.py
```

The last command starts an MCP stdio server, not an interactive terminal UI.
First use downloads the public `all-MiniLM-L6-v2` sentence-transformer model.
Document embeddings and the generated manifest stay in ignored `.data/`.
There are no paid API calls. The model is used for retrieval, not generation.

Configure an MCP client to run your virtual environment's Python executable
with the absolute path to `server.py` as its argument.

## Tools

`vault_search`, `get_note`, `get_service`, `get_node`, `whereis`, `backlinks`,
`list_services`, `list_nodes`, `get_manifest`, and `reindex`.

Markdown -> YAML/frontmatter and heading parsing -> content hashing and
incremental indexing -> normalized embeddings + FTS5 -> reciprocal rank fusion
-> source paths, headings, snippets, and full-note retrieval through MCP.

The retrieval, chunking, indexing, and MCP tools come from the original project.
The fictional vault, portable path defaults, containment checks, and automated
public tests were added for this edition. This repository demonstrates one of
the three MCP servers described in the resume, not all 32 tools.

## Your own documents

Set `SERVER_BRAIN_VAULT` explicitly to an approved document directory. Remove
the generated `.data/` directory before switching vaults. Never point it at
credentials, customer data, or a mixed private document store. Any connected
MCP client can read the configured vault; this local server is not a multiuser
authorization boundary. Retrieved text is untrusted input to an agent.

## Tests

```sh
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

Tests include real embedding retrieval over fictional documents and a real
stdio MCP handshake. First run requires a model download.
