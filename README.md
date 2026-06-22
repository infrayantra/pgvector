# pgvector Reference

A detailed reference for [pgvector](https://github.com/pgvector/pgvector) — the open-source PostgreSQL extension for vector similarity search.

## Contents

| Document | Description |
|----------|-------------|
| [REFERENCE.md](./REFERENCE.md) | Full guide: installation (all OS), SQL usage, indexing, performance, troubleshooting |
| [USECASES-AND-INDEXES.md](./USECASES-AND-INDEXES.md) | **Use cases** (RAG, search, recs, fraud, etc.) and **index types** (HNSW, IVFFlat, opclasses, tuning) |
| [INTEGRATION.md](./INTEGRATION.md) | Application integration: Python, Node.js, Go, Java, Rust, and framework examples |
| [demo/](./demo/) | **Interactive Python demo** — menu-driven use cases with Docker Postgres |

## Quick Start

```sql
-- 1. Enable extension (once per database)
CREATE EXTENSION vector;

-- 2. Create table with embeddings
CREATE TABLE documents (
  id bigserial PRIMARY KEY,
  content text,
  embedding vector(1536)
);

-- 3. Nearest-neighbor search (cosine distance)
SELECT id, content
FROM documents
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 5;
```

## Interactive Demo

```bash
cd demo
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Press `S` to start Docker, `I` to init, then pick use cases `1`–`9`. See [demo/README.md](./demo/README.md).

## Requirements

- PostgreSQL **13+**
- pgvector **0.8.x** (latest stable as of 2025)

## Official Resources

- [pgvector GitHub](https://github.com/pgvector/pgvector)
- [pgvector-python](https://github.com/pgvector/pgvector-python)
- [pgvector-node](https://github.com/pgvector/pgvector-node)
- [Docker images](https://hub.docker.com/r/pgvector/pgvector)
