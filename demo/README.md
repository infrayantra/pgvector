# pgvector Interactive Demo

Menu-driven Python demos for pgvector use cases, backed by a Docker Postgres container.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/Mac) or Docker Engine (Linux)
- Python 3.10+

## Quick Start

```bash
cd demo

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

pip install -r requirements.txt

# Run interactive menu
python main.py
```

### First-time workflow

1. Press **`S`** — start the pgvector Docker container
2. Press **`I`** — initialize the `vector` extension
3. Pick a use case **`1`–`9`** or run all with **`A`**

## Docker (manual)

```bash
cd demo
docker compose up -d
docker compose ps
docker compose logs -f
docker compose down        # stop
docker compose down -v   # stop + delete data
```

**Connection details:**

| Setting | Value |
|---------|-------|
| Host | `localhost` |
| Port | `5433` (avoids conflict with local Postgres on 5432) |
| Database | `pgvector_demo` |
| User | `pgvector` |
| Password | `pgvector` |

```bash
# Connect with psql
psql postgresql://pgvector:pgvector@localhost:5433/pgvector_demo
```

## Use Cases

| # | Demo | What it shows |
|---|------|---------------|
| 1 | Basic vector search | INSERT, exact k-NN, query plan |
| 2 | Semantic document search | HNSW + cosine similarity over text |
| 3 | RAG retrieval | Chunked knowledge base, context assembly |
| 4 | Hybrid search | Full-text (GIN) + vector + RRF fusion |
| 5 | Recommendations | Item similarity, user profile averaging |
| 6 | Deduplication | Distance threshold, near-duplicate pairs |
| 7 | Filtered search | Metadata filters + iterative HNSW scan |
| 8 | Distance metrics | L2 vs cosine vs inner product |
| 9 | HNSW vs exact | Latency and recall at different `ef_search` |

## Embeddings

By default the demo uses **SentenceTransformers** (`all-MiniLM-L6-v2`, 384 dimensions).

- First run downloads ~90MB model weights
- If `sentence-transformers` is not installed, a deterministic hash fallback is used (semantic quality reduced)

To skip the ML model:

```bash
pip install psycopg[binary] pgvector numpy
```

## Project layout

```
demo/
├── docker-compose.yml    # pgvector/pgvector:pg16 on port 5433
├── main.py               # Menu-driven entry point
├── db.py                 # Connection, schema helpers
├── embeddings.py         # SentenceTransformers / fallback
├── requirements.txt
└── use_cases/
    ├── basic_search.py
    ├── semantic_search.py
    ├── rag.py
    ├── hybrid_search.py
    ├── recommendations.py
    ├── deduplication.py
    ├── filtering.py
    ├── distance_metrics.py
    └── index_comparison.py
```

## Environment

Copy `.env.example` to `.env` to override the database URL:

```
DATABASE_URL=postgresql://pgvector:pgvector@localhost:5433/pgvector_demo
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Connection refused` on 5433 | Run menu option `[S]` or `docker compose up -d` |
| `extension "vector" is not available` | Use the pgvector Docker image (not plain `postgres`) |
| Slow first embedding call | Model download; subsequent runs are fast |
| Port 5433 in use | Change port in `docker-compose.yml` and `.env` |
