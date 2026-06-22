# pgvector Complete Reference

> **Version note:** This reference targets pgvector **0.8.x** on PostgreSQL **13–18**. Replace version numbers in package names with your installed PostgreSQL major version.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Prerequisites](#2-prerequisites)
3. [Installation by Operating System](#3-installation-by-operating-system)
4. [Enabling the Extension](#4-enabling-the-extension)
5. [Data Types](#5-data-types)
6. [Storing Vectors](#6-storing-vectors)
7. [Querying & Distance Functions](#7-querying--distance-functions)
8. [Indexing (HNSW & IVFFlat)](#8-indexing-hnsw--ivfflat)
9. [Filtering & Hybrid Search](#9-filtering--hybrid-search)
10. [Advanced Features](#10-advanced-features)
11. [Performance & Scaling](#11-performance--scaling)
12. [Monitoring & Troubleshooting](#12-monitoring--troubleshooting)
13. [Upgrading](#13-upgrading)
14. [SQL API Reference](#14-sql-api-reference)
15. [Hosted Providers](#15-hosted-providers)

---

## 1. Overview

**pgvector** adds vector similarity search to PostgreSQL. Store embeddings alongside relational data and query them with standard SQL.

### Key capabilities

| Feature | Details |
|---------|---------|
| Search modes | Exact nearest neighbor; approximate (HNSW, IVFFlat) |
| Vector types | `vector`, `halfvec`, `bit`, `sparsevec` |
| Distance metrics | L2, inner product, cosine, L1, Hamming, Jaccard |
| Dimensions | Up to 16,000 (`vector`); indexable up to 2,000 (4,000 with `halfvec`) |
| Postgres features | ACID, WAL/replication, JOINs, partitioning, full-text search |

### When to use pgvector

- **RAG** (retrieval-augmented generation) document stores
- Semantic search over text, images, or structured data
- Recommendation systems
- Deduplication and clustering
- Any workload where vectors live with other relational data

---

## 2. Prerequisites

| Requirement | Notes |
|-------------|-------|
| PostgreSQL 13+ | Extension supports PG 13 through 18 |
| Superuser or `CREATE` on database | Needed for `CREATE EXTENSION` |
| Build tools (source install) | `gcc`/`clang`, `make`, PostgreSQL dev headers |
| Windows | Visual Studio with C++ workload, `nmake` |

### Verify PostgreSQL version

```bash
psql -c "SELECT version();"
# or
pg_config --version
```

---

## 3. Installation by Operating System

Installation puts extension binaries on disk. **Enabling** (`CREATE EXTENSION`) is a separate per-database step (see [Section 4](#4-enabling-the-extension)).

### 3.1 Ubuntu / Debian (APT — recommended)

Use the [PostgreSQL APT repository](https://wiki.postgresql.org/wiki/Apt).

```bash
# One-time: add PGDG repo (see wiki for your distro/version)
sudo apt update
sudo apt install postgresql-18-pgvector   # replace 18 with your PG major
```

**Package naming:** `postgresql-{MAJOR}-pgvector`

| PostgreSQL | Package |
|------------|---------|
| 16 | `postgresql-16-pgvector` |
| 17 | `postgresql-17-pgvector` |
| 18 | `postgresql-18-pgvector` |

### 3.2 RHEL / Rocky / AlmaLinux / Fedora (Yum/DNF)

Use the [PostgreSQL Yum repository](https://yum.postgresql.org/).

```bash
sudo dnf install pgvector_18    # replace 18 with your PG major
# or
sudo yum install pgvector_18
```

**Package naming:** `pgvector_{MAJOR}`

### 3.3 macOS

#### Homebrew (easiest)

```bash
brew install pgvector
```

> Adds pgvector to `postgresql@17` and `postgresql@18` Homebrew formulas.

#### Postgres.app

Download [Postgres.app](https://postgresapp.com/) (PostgreSQL 15+). pgvector is **preinstalled**.

#### Compile from source

```bash
cd /tmp
git clone --branch v0.8.2 https://github.com/pgvector/pgvector.git
cd pgvector
export PG_CONFIG=/opt/homebrew/opt/postgresql@18/bin/pg_config   # adjust path
make
make install   # may need sudo
```

**Common `pg_config` paths on macOS:**

| Install method | Path |
|----------------|------|
| EDB installer | `/Library/PostgreSQL/18/bin/pg_config` |
| Homebrew (arm64) | `/opt/homebrew/opt/postgresql@18/bin/pg_config` |
| Homebrew (x86) | `/usr/local/opt/postgresql@18/bin/pg_config` |

**Multiple Postgres installs:**

```bash
export PG_CONFIG=/path/to/pg_config
make clean && make
sudo --preserve-env=PG_CONFIG make install
```

### 3.4 Windows

#### Option A: Build from source

1. Install [Visual Studio](https://visualstudio.microsoft.com/) with **Desktop development with C++**
2. Open **x64 Native Tools Command Prompt for VS** as Administrator
3. Build:

```cmd
set "PGROOT=C:\Program Files\PostgreSQL\18"
cd %TEMP%
git clone --branch v0.8.2 https://github.com/pgvector/pgvector.git
cd pgvector
nmake /F Makefile.win
nmake /F Makefile.win install
```

**Windows troubleshooting:**

| Error | Fix |
|-------|-----|
| `Cannot open include file: 'postgres.h'` | Verify `PGROOT` points to correct PostgreSQL install |
| `case value '4' already used` | Use x64 Native Tools prompt; run `nmake /F Makefile.win clean` |
| `unresolved external symbol float_to_shortest_decimal_bufn` (PG 17.0–17.2) | Upgrade to PostgreSQL 17.3+ |
| `Access is denied` | Run as Administrator |

#### Option B: Docker (recommended on Windows)

```bash
docker pull pgvector/pgvector:pg18
docker run -d --name pgvector -e POSTGRES_PASSWORD=secret -p 5432:5432 pgvector/pgvector:pg18
```

#### Option C: conda-forge

```bash
conda install -c conda-forge pgvector
```

### 3.5 Docker

Official image extends the [Postgres Docker image](https://hub.docker.com/_/postgres).

```bash
docker pull pgvector/pgvector:pg18-trixie
```

**Run with persistent volume:**

```bash
docker run -d \
  --name pgvector \
  -e POSTGRES_PASSWORD=yourpassword \
  -e POSTGRES_DB=vectordb \
  -p 5432:5432 \
  -v pgdata:/var/lib/postgresql/data \
  --shm-size=1g \
  pgvector/pgvector:pg18
```

> Use `--shm-size` ≥ `maintenance_work_mem` when building large HNSW indexes in parallel.

**Supported tags (examples):**

| Tag | Description |
|-----|-------------|
| `pg18`, `pg18-bookworm` | PostgreSQL 18 |
| `pg17`, `pg17-trixie` | PostgreSQL 17 |
| `pg16`, `pg16-bookworm` | PostgreSQL 16 |
| `0.8.2-pg18` | Pinned pgvector + PG version |

**Build custom image:**

```bash
git clone --branch v0.8.2 https://github.com/pgvector/pgvector.git
cd pgvector
docker build --pull --build-arg PG_MAJOR=18 -t myuser/pgvector .
```

**docker-compose example:**

```yaml
services:
  db:
    image: pgvector/pgvector:pg18
    environment:
      POSTGRES_USER: app
      POSTGRES_PASSWORD: secret
      POSTGRES_DB: vectordb
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    shm_size: 1gb

volumes:
  pgdata:
```

### 3.6 Alpine Linux (APK)

```bash
apk add postgresql-pgvector
```

### 3.7 FreeBSD (pkg / ports)

```bash
pkg install postgresql17-pgvector
```

Or from ports:

```bash
cd /usr/ports/databases/pgvector
make install
```

### 3.8 PGXN (any platform with PGXN client)

```bash
pgxn install vector
```

### 3.9 Compile from source (Linux / macOS)

```bash
# Install dev headers first (Ubuntu/Debian):
sudo apt install postgresql-server-dev-18 build-essential

cd /tmp
git clone --branch v0.8.2 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

**Portability build** (avoid `-march=native` issues when moving binaries):

```bash
make OPTFLAGS=""
```

**Source install troubleshooting:**

| Error | Fix |
|-------|-----|
| `fatal error: postgres.h: No such file` | Install `postgresql-server-dev-{MAJOR}` |
| `no such sysroot directory` (macOS) | Reinstall PostgreSQL; check `pg_config --cppflags` |
| `Illegal instruction` at runtime | Rebuild with `make OPTFLAGS=""` |

### 3.10 GitHub Actions (CI)

Use [setup-pgvector](https://github.com/pgvector/setup-pgvector) action in workflows.

### 3.11 Installation method comparison

| Method | Best for | Pros | Cons |
|--------|----------|------|------|
| APT/Yum package | Production Linux | No compile; version-matched | Needs PGDG repo |
| Homebrew | macOS dev | One command | Limited PG versions |
| Docker | All platforms, CI | Isolated, reproducible | Container overhead |
| Source | Custom PG builds | Full control | Requires toolchain |
| Hosted DB | Managed prod | Zero ops | Vendor lock-in |

---

## 4. Enabling the Extension

Installation ≠ enablement. Run **once per database** that needs vectors.

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Verify installation

```sql
-- Extension version
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';

-- Available functions
SELECT proname FROM pg_proc
WHERE pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'public')
  AND proname LIKE '%distance%';

-- In psql
\dx vector
```

Expected functions include: `l2_distance`, `cosine_distance`, `inner_product`, `vector_dims`, `vector_norm`.

### Common enablement errors

| Error | Cause | Fix |
|-------|-------|-----|
| `extension "vector" is not available` | Binary not installed for this PG version | Install matching package or rebuild |
| `could not open extension control file` | Wrong PG instance / version mismatch | Align package with running `postgres` version |
| `type "vector" does not exist` | Extension not enabled in this DB | `CREATE EXTENSION vector;` |
| Extension in wrong schema | Non-default `search_path` | `SET search_path TO public;` or qualify types |

### Permissions

```sql
-- Grant to application role
GRANT USAGE ON SCHEMA public TO app_user;
-- Extension objects live in the extension's schema (usually public)
```

---

## 5. Data Types

### 5.1 `vector` (primary type)

Single-precision float32 array. Storage: `4 × dimensions + 8` bytes.

```sql
CREATE TABLE items (
  id bigserial PRIMARY KEY,
  embedding vector(1536)   -- fixed dimensions (recommended)
);

-- Variable dimensions (no index across mixed sizes without tricks)
CREATE TABLE embeddings (
  model_id bigint,
  item_id bigint,
  embedding vector,
  PRIMARY KEY (model_id, item_id)
);
```

**Limits:** Up to 16,000 dimensions; HNSW/IVFFlat index up to 2,000 dimensions.

**Input format:** `'[1,2,3]'` or `'[1.0, 2.0, 3.0]'`

### 5.2 `halfvec`

Half-precision (float16). Half the memory of `vector`.

```sql
CREATE TABLE items (embedding halfvec(1536));
```

Storage: `2 × dimensions + 8` bytes. Indexable up to 4,000 dimensions.

### 5.3 `bit` (binary vectors)

For Hamming / Jaccard distance on binary data.

```sql
CREATE TABLE items (embedding bit(128));
INSERT INTO items (embedding) VALUES ('101010...');
```

### 5.4 `sparsevec`

Sparse vectors: `{index:value,...}/dimensions` (indices start at 1).

```sql
CREATE TABLE items (embedding sparsevec(10000));
INSERT INTO items (embedding) VALUES ('{1:1,3:2,5:3}/5');
```

### 5.5 Higher precision storage

Use `double precision[]` or `numeric[]` when you need more precision than float32, then cast for search:

```sql
CREATE TABLE items (embedding double precision[]);
INSERT INTO items (embedding) VALUES ('{1,2,3}');

ALTER TABLE items ADD CHECK (vector_dims(embedding::vector) = 3);

-- Query with cast
SELECT * FROM items ORDER BY embedding::vector(3) <-> '[1,2,3]' LIMIT 5;
```

---

## 6. Storing Vectors

### Create tables

```sql
CREATE TABLE documents (
  id bigserial PRIMARY KEY,
  title text NOT NULL,
  content text NOT NULL,
  metadata jsonb DEFAULT '{}',
  embedding vector(1536),
  created_at timestamptz DEFAULT now()
);

-- Add column to existing table
ALTER TABLE articles ADD COLUMN embedding vector(768);
```

### Insert

```sql
INSERT INTO documents (title, content, embedding)
VALUES (
  'Hello World',
  'Sample document text',
  '[0.1, 0.2, 0.3, ...]'::vector
);

-- Batch insert
INSERT INTO documents (title, content, embedding) VALUES
  ('Doc A', 'Text A', '[...]'::vector),
  ('Doc B', 'Text B', '[...]'::vector);
```

### Upsert

```sql
INSERT INTO documents (id, title, content, embedding)
VALUES (1, 'Updated', 'New text', '[...]'::vector)
ON CONFLICT (id) DO UPDATE SET
  title = EXCLUDED.title,
  content = EXCLUDED.content,
  embedding = EXCLUDED.embedding;
```

### Update / Delete

```sql
UPDATE documents SET embedding = '[...]'::vector WHERE id = 1;
DELETE FROM documents WHERE id = 1;
```

### Bulk load with COPY

Fastest method for large datasets. Load data first, then create indexes.

```sql
COPY documents (id, title, content, embedding)
FROM '/path/to/data.csv' WITH (FORMAT csv, HEADER true);
```

Binary COPY is supported from application clients (see [INTEGRATION.md](./INTEGRATION.md)).

### Normalize embeddings (cosine similarity workflows)

```sql
UPDATE documents SET embedding = l2_normalize(embedding);
```

OpenAI and many models output L2-normalized vectors — use **inner product** (`<#>`) for best performance with normalized vectors.

---

## 7. Querying & Distance Functions

### Distance operators

| Operator | Metric | Index opclass | Notes |
|----------|--------|---------------|-------|
| `<->` | L2 (Euclidean) | `vector_l2_ops` | Default choice |
| `<#>` | Negative inner product | `vector_ip_ops` | Use for normalized vectors |
| `<=>` | Cosine distance | `vector_cosine_ops` | `1 - cosine_similarity` |
| `<+>` | L1 (Manhattan) | `vector_l1_ops` | |
| `<~>` | Hamming | `bit_hamming_ops` | Binary vectors |
| `<%>` | Jaccard | `bit_jaccard_ops` | Binary vectors |

> `<#>` returns **negative** inner product because Postgres index scans require ascending `ORDER BY`.

### Nearest neighbors (k-NN)

```sql
-- L2 distance
SELECT id, title, embedding <-> '[...]'::vector AS distance
FROM documents
ORDER BY embedding <-> '[...]'::vector
LIMIT 10;

-- Cosine distance
SELECT id, title, 1 - (embedding <=> '[...]'::vector) AS similarity
FROM documents
ORDER BY embedding <=> '[...]'::vector
LIMIT 10;

-- Inner product (normalized vectors)
SELECT id, title, (embedding <#> '[...]'::vector) * -1 AS inner_product
FROM documents
ORDER BY embedding <#> '[...]'::vector
LIMIT 10;
```

### Find neighbors of an existing row

```sql
SELECT d2.*
FROM documents d1
JOIN documents d2 ON d1.id != d2.id
WHERE d1.id = 42
ORDER BY d2.embedding <-> d1.embedding
LIMIT 5;
```

### Distance threshold

```sql
SELECT * FROM documents
WHERE embedding <-> '[...]'::vector < 0.5
ORDER BY embedding <-> '[...]'::vector
LIMIT 20;
```

> Combine `WHERE distance` + `ORDER BY` + `LIMIT` for index use with iterative scans (see [Section 8](#8-indexing-hnsw--ivfflat)).

### Aggregates

```sql
SELECT AVG(embedding) FROM documents;
SELECT category, AVG(embedding) FROM documents GROUP BY category;
SELECT SUM(embedding) FROM documents;
```

### Choosing a distance metric

| Embedding source | Recommended metric | Operator |
|------------------|-------------------|----------|
| OpenAI `text-embedding-3-*` | Inner product (vectors normalized) | `<#>` |
| Sentence-BERT (normalized) | Cosine or inner product | `<=>` or `<#>` |
| Raw feature vectors | L2 | `<->` |
| Binary / perceptual hashes | Hamming | `<~>` |

---

## 8. Indexing (HNSW & IVFFlat)

By default, pgvector performs **exact** search (perfect recall, slower at scale).

Approximate indexes trade recall for speed. **Results may differ** from exact search after adding an index.

### 8.1 HNSW (recommended for most workloads)

Hierarchical Navigable Small World graph. Better query performance than IVFFlat; slower builds; more memory. Can be created on empty tables.

```sql
-- L2
CREATE INDEX documents_embedding_idx ON documents
  USING hnsw (embedding vector_l2_ops);

-- Cosine
CREATE INDEX documents_embedding_cosine_idx ON documents
  USING hnsw (embedding vector_cosine_ops);

-- Inner product
CREATE INDEX documents_embedding_ip_idx ON documents
  USING hnsw (embedding vector_ip_ops);

-- With parameters
CREATE INDEX documents_embedding_idx ON documents
  USING hnsw (embedding vector_l2_ops)
  WITH (m = 16, ef_construction = 64);

-- Non-blocking (production)
CREATE INDEX CONCURRENTLY documents_embedding_idx ON documents
  USING hnsw (embedding vector_l2_ops);
```

**HNSW parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `m` | 16 | Max connections per layer (higher = better recall, more memory) |
| `ef_construction` | 64 | Build-time candidate list (higher = better recall, slower build) |
| `hnsw.ef_search` | 40 | Query-time candidate list (session setting) |

```sql
SET hnsw.ef_search = 100;   -- higher recall, slower queries

BEGIN;
SET LOCAL hnsw.ef_search = 100;
SELECT ... ORDER BY embedding <=> '[...]' LIMIT 10;
COMMIT;
```

**HNSW build tuning:**

```sql
SET maintenance_work_mem = '8GB';
SET max_parallel_maintenance_workers = 7;
CREATE INDEX CONCURRENTLY ...;
```

Monitor build progress:

```sql
SELECT phase, round(100.0 * blocks_done / nullif(blocks_total, 0), 1) AS pct
FROM pg_stat_progress_create_index;
```

### 8.2 IVFFlat

Inverted file with flat compression. Faster builds, less memory, lower query performance.

**Requires data before index creation** (training step).

```sql
CREATE INDEX documents_embedding_ivfflat_idx ON documents
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);
```

**Choosing `lists`:**

| Rows | Suggested `lists` |
|------|-------------------|
| < 1M | `rows / 1000` |
| ≥ 1M | `sqrt(rows)` |

**Query probes:**

```sql
SET ivfflat.probes = 10;   -- start with sqrt(lists)

BEGIN;
SET LOCAL ivfflat.probes = 10;
SELECT ... LIMIT 10;
COMMIT;
```

### 8.3 HNSW vs IVFFlat

| | HNSW | IVFFlat |
|---|------|---------|
| Build time | Slower | Faster |
| Memory | Higher | Lower |
| Query speed | Faster | Slower |
| Needs data first? | No | Yes (for good recall) |
| Best for | Production ANN | Large scale, memory-constrained |

### 8.4 Index requirements for query planner

For the planner to use an ANN index:

```sql
-- ✓ Uses index
ORDER BY embedding <=> '[...]'::vector LIMIT 5;

-- ✗ No index (expression in ORDER BY)
ORDER BY 1 - (embedding <=> '[...]'::vector) DESC LIMIT 5;
```

Force index (debugging only):

```sql
BEGIN;
SET LOCAL enable_seqscan = off;
SELECT ... ORDER BY embedding <=> '[...]' LIMIT 5;
COMMIT;
```

### 8.5 Iterative index scans (pgvector 0.8+)

When filtering with `WHERE`, approximate indexes may return too few rows. Enable iterative scans:

```sql
SET hnsw.iterative_scan = strict_order;   -- exact distance ordering
-- or
SET hnsw.iterative_scan = relaxed_order;  -- better recall, slight reordering
SET ivfflat.iterative_scan = relaxed_order;

SET hnsw.max_scan_tuples = 20000;
SET hnsw.scan_mem_multiplier = 2;
SET ivfflat.max_probes = 100;
```

**Filtered k-NN with materialized CTE:**

```sql
WITH nearest AS MATERIALIZED (
  SELECT id, content, embedding <=> '[...]'::vector AS distance
  FROM documents
  WHERE category = 'tech'
  ORDER BY distance
  LIMIT 10
)
SELECT * FROM nearest WHERE distance < 0.3 ORDER BY distance;
```

---

## 9. Filtering & Hybrid Search

### Metadata filtering

```sql
SELECT id, title
FROM documents
WHERE metadata->>'source' = 'wiki'
ORDER BY embedding <=> '[...]'::vector
LIMIT 10;
```

**Strategies:**

1. **B-tree on filter column** — good when filter is selective
2. **Partial HNSW index** — fixed filter values
3. **Table partitioning** — many distinct filter values
4. **Iterative scans** — general-purpose with ANN indexes

```sql
CREATE INDEX ON documents (category_id);
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops)
  WHERE category_id = 123;
```

### Full-text + vector (hybrid search)

```sql
-- Keyword search
SELECT id, content,
  ts_rank_cd(textsearch, query) AS text_rank
FROM documents, plainto_tsquery('machine learning') query
WHERE textsearch @@ query
ORDER BY text_rank DESC
LIMIT 20;

-- Combine with vector search using Reciprocal Rank Fusion (RRF) in application code
-- score = sum(1 / (k + rank_i)) for each ranking method
```

Add GIN index for full-text:

```sql
ALTER TABLE documents ADD COLUMN textsearch tsvector
  GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;
CREATE INDEX ON documents USING GIN (textsearch);
```

---

## 10. Advanced Features

### Half-precision indexing

```sql
CREATE INDEX ON documents
  USING hnsw ((embedding::halfvec(1536)) halfvec_cosine_ops);

SELECT * FROM documents
ORDER BY embedding::halfvec(1536) <=> '[...]'::halfvec
LIMIT 10;
```

### Binary quantization (large scale)

```sql
CREATE INDEX ON documents
  USING hnsw ((binary_quantize(embedding)::bit(1536)) bit_hamming_ops);

-- Two-stage: coarse Hamming, then re-rank with full vectors
SELECT * FROM (
  SELECT * FROM documents
  ORDER BY binary_quantize(embedding)::bit(1536) <~> binary_quantize('[...]'::vector)
  LIMIT 50
) sub
ORDER BY embedding <=> '[...]'::vector
LIMIT 10;
```

### Subvector indexing

```sql
CREATE INDEX ON documents
  USING hnsw ((subvector(embedding, 1, 256)::vector(256)) vector_cosine_ops);
```

### Multi-model embeddings

```sql
CREATE INDEX ON embeddings
  USING hnsw ((embedding::vector(1536)) vector_cosine_ops)
  WHERE model_id = 1;

SELECT * FROM embeddings
WHERE model_id = 1
ORDER BY embedding::vector(1536) <=> '[...]'::vector(1536)
LIMIT 5;
```

---

## 11. Performance & Scaling

### Postgres tuning

Use [PgTune](https://pgtune.leopard.in.ua/) for initial values.

```sql
SHOW config_file;
SHOW shared_buffers;   -- typically 25% of RAM
```

Key settings:

| Setting | Purpose |
|---------|---------|
| `shared_buffers` | Cache shared data |
| `maintenance_work_mem` | Index builds (especially HNSW) |
| `work_mem` | Sort/hash per operation |
| `max_parallel_workers_per_gather` | Parallel exact search |
| `effective_cache_size` | Planner hint for OS cache |

### Loading best practices

1. Bulk load with `COPY`
2. Create indexes **after** initial load
3. Use `CREATE INDEX CONCURRENTLY` in production
4. Use `halfvec` for smaller working set
5. Use binary quantization + re-rank for very large indexes

### Query debugging

```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM documents
ORDER BY embedding <=> '[...]'::vector
LIMIT 10;
```

### Vertical vs horizontal scaling

| Approach | Method |
|----------|--------|
| Vertical | More RAM/CPU; tune `shared_buffers`, index in memory |
| Read replicas | Offload search reads to standbys |
| Sharding | [Citus](https://github.com/citusdata/citus), [PgDog](https://github.com/pgdogdev/pgdog) |

### Vacuuming HNSW indexes

```sql
REINDEX INDEX CONCURRENTLY documents_embedding_idx;
VACUUM documents;
```

---

## 12. Monitoring & Troubleshooting

### Check index size

```sql
SELECT pg_size_pretty(pg_relation_size('documents_embedding_idx'));
```

### Compare approximate vs exact recall

```sql
BEGIN;
SET LOCAL enable_indexscan = off;  -- force exact search
SELECT ... ORDER BY embedding <=> '[...]' LIMIT 10;
COMMIT;
```

### Common issues

| Symptom | Likely cause | Solution |
|---------|--------------|----------|
| Query not using index | Missing `ORDER BY` + `LIMIT`; expression in ORDER BY | Fix query shape |
| Fewer results after HNSW | Low `ef_search`; filtering | Raise `hnsw.ef_search`; enable iterative scan |
| Fewer results after IVFFlat | Too few lists; low probes | Drop index, add data, rebuild; raise `ivfflat.probes` |
| Slow exact search | Large table, no index | Add HNSW or raise `max_parallel_workers_per_gather` |
| NULL vectors missing | NULLs not indexed | Filter `WHERE embedding IS NOT NULL` |
| Zero vectors (cosine) | Not indexed | Normalize or filter |

### pg_stat_statements

```sql
CREATE EXTENSION pg_stat_statements;
SELECT query, calls, mean_exec_time
FROM pg_stat_statements
WHERE query LIKE '%<=>%'
ORDER BY mean_exec_time DESC
LIMIT 10;
```

---

## 13. Upgrading

```bash
# Reinstall using same method (package, source, etc.)
sudo apt install --only-upgrade postgresql-18-pgvector
```

Per database:

```sql
ALTER EXTENSION vector UPDATE;
SELECT extversion FROM pg_extension WHERE extname = 'vector';
```

---

## 14. SQL API Reference

### Vector operators

| Operator | Description |
|----------|-------------|
| `+` | Element-wise addition |
| `-` | Element-wise subtraction |
| `*` | Element-wise multiplication |
| `\|\|` | Concatenate |

### Vector functions

| Function | Returns | Description |
|----------|---------|-------------|
| `l2_distance(a, b)` | `float8` | Euclidean distance |
| `inner_product(a, b)` | `float8` | Inner product |
| `cosine_distance(a, b)` | `float8` | Cosine distance |
| `l1_distance(a, b)` | `float8` | Manhattan distance |
| `vector_dims(v)` | `int` | Number of dimensions |
| `vector_norm(v)` | `float8` | L2 norm |
| `l2_normalize(v)` | `vector` | Unit vector |
| `subvector(v, start, count)` | `vector` | Slice subvector |
| `binary_quantize(v)` | `bit` | Binary quantization |

### Aggregates

| Function | Description |
|----------|-------------|
| `avg(vector)` | Average vector |
| `sum(vector)` | Sum of vectors |

---

## 15. Hosted Providers

pgvector is available on many managed Postgres services including:

- **AWS RDS / Aurora**
- **Google Cloud SQL / AlloyDB**
- **Azure Database for PostgreSQL**
- **Supabase**
- **Neon**
- **Crunchy Bridge**
- **Timescale**
- **Railway**, **Render**, **Fly.io**

Enable with `CREATE EXTENSION vector;` (or provider dashboard). Check provider docs for version limits and instance sizing for vector workloads.

---

## Appendix: End-to-end RAG schema example

```sql
CREATE EXTENSION vector;

CREATE TABLE knowledge_base (
  id bigserial PRIMARY KEY,
  source text NOT NULL,
  chunk_index int NOT NULL,
  content text NOT NULL,
  textsearch tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
  embedding vector(1536) NOT NULL,
  metadata jsonb DEFAULT '{}',
  created_at timestamptz DEFAULT now(),
  UNIQUE (source, chunk_index)
);

CREATE INDEX knowledge_base_embedding_idx ON knowledge_base
  USING hnsw (embedding vector_cosine_ops);

CREATE INDEX knowledge_base_textsearch_idx ON knowledge_base
  USING GIN (textsearch);

CREATE INDEX knowledge_base_metadata_idx ON knowledge_base
  USING GIN (metadata);

-- Semantic search
SELECT id, content, embedding <=> $1::vector AS distance
FROM knowledge_base
WHERE metadata->>'tenant_id' = $2
ORDER BY distance
LIMIT 10;
```

See [INTEGRATION.md](./INTEGRATION.md) for application-layer embedding generation and client library usage.
