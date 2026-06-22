# pgvector Use Cases & Index Types — Deep Dive

A comprehensive guide to **what pgvector is used for**, **which index to choose**, and **how every indexing option works** in practice.

> See also: [REFERENCE.md](./REFERENCE.md) (installation & SQL) · [INTEGRATION.md](./INTEGRATION.md) (app code)

---

## Table of Contents

1. [Use Case Catalog](#1-use-case-catalog)
2. [Index Types Overview](#2-index-types-overview)
3. [Exact Search (No Vector Index)](#3-exact-search-no-vector-index)
4. [HNSW Index — Full Reference](#4-hnsw-index--full-reference)
5. [IVFFlat Index — Full Reference](#5-ivfflat-index--full-reference)
6. [Operator Classes (Opclasses)](#6-operator-classes-opclasses)
7. [Expression & Specialized Indexes](#7-expression--specialized-indexes)
8. [Combining Vector Indexes with Relational Indexes](#8-combining-vector-indexes-with-relational-indexes)
9. [Index Selection Decision Guide](#9-index-selection-decision-guide)
10. [Distance Metrics by Use Case](#10-distance-metrics-by-use-case)
11. [Vector Data Types by Use Case](#11-vector-data-types-by-use-case)
12. [Scale & Capacity Planning](#12-scale--capacity-planning)
13. [pgvector vs Dedicated Vector Databases](#13-pgvector-vs-dedicated-vector-databases)
14. [Anti-Patterns & Common Mistakes](#14-anti-patterns--common-mistakes)

---

## 1. Use Case Catalog

### 1.1 Retrieval-Augmented Generation (RAG)

**What:** Store document chunks as embeddings; at query time retrieve the most relevant chunks and pass them to an LLM as context.

**Why pgvector:** Vectors live next to metadata (`source`, `page`, `tenant_id`, `permissions`), enabling filtered retrieval in one SQL query.

```sql
CREATE TABLE rag_chunks (
  id bigserial PRIMARY KEY,
  document_id uuid NOT NULL,
  chunk_index int NOT NULL,
  content text NOT NULL,
  token_count int,
  embedding vector(1536),
  metadata jsonb DEFAULT '{}',
  created_at timestamptz DEFAULT now()
);

CREATE INDEX rag_chunks_hnsw ON rag_chunks
  USING hnsw (embedding vector_cosine_ops);

-- Tenant-scoped retrieval
SELECT content, embedding <=> $1::vector AS distance
FROM rag_chunks
WHERE metadata->>'tenant_id' = $2
ORDER BY distance LIMIT 8;
```

| Detail | Recommendation |
|--------|----------------|
| Embedding model | `text-embedding-3-small` (1536d) or domain-specific |
| Chunk size | 256–1024 tokens; 10–20% overlap |
| Index | HNSW + `vector_cosine_ops` |
| k | 5–20 chunks per query |
| Hybrid | Add GIN on `tsvector` for keyword + vector RRF |

---

### 1.2 Semantic / Neural Search

**What:** Search by meaning, not keywords. "automobile repair" matches "car maintenance."

```sql
CREATE TABLE products (
  id serial PRIMARY KEY,
  name text,
  description text,
  category_id int,
  embedding vector(384)   -- e.g. all-MiniLM-L6-v2
);

CREATE INDEX products_semantic ON products
  USING hnsw (embedding vector_cosine_ops);
CREATE INDEX products_category ON products (category_id);
```

**Use when:** Users write natural-language queries; catalog descriptions are rich text.

---

### 1.3 Hybrid Search (Keyword + Semantic)

**What:** Combine PostgreSQL full-text search (BM25/ts_rank) with vector similarity.

```sql
-- Vector leg
SELECT id, row_number() OVER (ORDER BY embedding <=> $1) AS rank
FROM articles WHERE embedding <=> $1 < 0.5 LIMIT 50;

-- Keyword leg
SELECT id, row_number() OVER (ORDER BY ts_rank_cd(textsearch, query) DESC) AS rank
FROM articles, plainto_tsquery('english', $2) query
WHERE textsearch @@ query LIMIT 50;

-- Fuse in app: RRF score = Σ 1/(k + rank_i), k=60
```

| Index needed | Type |
|--------------|------|
| `embedding` | HNSW `vector_cosine_ops` |
| `textsearch` | GIN |
| `metadata` filters | B-tree or GIN (jsonb) |

---

### 1.4 Recommendation Systems

**What:** "Users who liked X also liked Y" — find items closest to a user's preference vector or an item's embedding.

```sql
CREATE TABLE user_preferences (
  user_id bigint PRIMARY KEY,
  preference_vector vector(128)   -- learned or averaged from history
);

CREATE TABLE items (
  id bigserial PRIMARY KEY,
  title text,
  item_vector vector(128)
);

CREATE INDEX items_rec ON items USING hnsw (item_vector vector_ip_ops);

-- Recommend for user
SELECT i.id, i.title
FROM items i, user_preferences u
WHERE u.user_id = $1
ORDER BY i.item_vector <#> u.preference_vector   -- inner product if normalized
LIMIT 20;
```

| Pattern | Details |
|---------|---------|
| Collaborative filtering | Store user/item latent vectors |
| Content-based | Embed item descriptions |
| Session-based | Average last N interaction embeddings |
| Metric | Inner product for normalized vectors |

---

### 1.5 Image & Multimodal Search

**What:** CLIP, SigLIP, or custom CNN embeddings for visual similarity.

```sql
CREATE TABLE images (
  id bigserial PRIMARY KEY,
  url text,
  label text,
  embedding vector(512)   -- CLIP ViT-B/32
);

CREATE INDEX images_hnsw ON images USING hnsw (embedding vector_cosine_ops);
```

**Binary / perceptual hash variant** (smaller, faster):

```sql
CREATE TABLE image_hashes (
  id bigserial PRIMARY KEY,
  phash bit(64)
);
CREATE INDEX image_hashes_hnsw ON image_hashes
  USING hnsw (phash bit_hamming_ops);
```

---

### 1.6 Audio & Video Similarity

**What:** Embed audio segments (Wav2Vec, Whisper) or video frames; find duplicates or similar clips.

```sql
CREATE TABLE audio_clips (
  id uuid PRIMARY KEY,
  duration_ms int,
  transcript text,
  audio_embedding vector(768),
  transcript_embedding vector(1536)
);
```

Often needs **two indexes** (one per embedding column) or a fused embedding.

---

### 1.7 Deduplication & Near-Duplicate Detection

**What:** Find documents/images within distance ε of each other.

```sql
-- Exact (small datasets)
SELECT a.id, b.id, a.embedding <-> b.embedding AS dist
FROM documents a
JOIN documents b ON a.id < b.id
WHERE a.embedding <-> b.embedding < 0.05;

-- At scale: LSH-style with binary quantization index + re-rank
SELECT * FROM (
  SELECT id FROM documents
  ORDER BY binary_quantize(embedding)::bit(1536) <~> binary_quantize($1::vector)
  LIMIT 100
) candidates
WHERE embedding <=> $1::vector < 0.05;
```

---

### 1.8 Anomaly Detection

**What:** Flag records far from cluster centroids or a "normal" reference set.

```sql
WITH centroid AS (
  SELECT AVG(embedding) AS center FROM normal_samples
)
SELECT s.id, s.embedding <-> c.center AS anomaly_score
FROM samples s, centroid c
ORDER BY anomaly_score DESC
LIMIT 100;
```

Usually **exact search** or HNSW with high `ef_search` for precision.

---

### 1.9 Fraud & Security (Behavioral Embeddings)

**What:** Embed session behavior, transaction patterns, or login fingerprints; compare to known-fraud vectors.

```sql
CREATE TABLE sessions (
  session_id uuid PRIMARY KEY,
  user_id bigint,
  behavior_embedding vector(256),
  risk_score float,
  flagged bool DEFAULT false
);

CREATE INDEX sessions_behavior ON sessions
  USING hnsw (behavior_embedding vector_l2_ops)
  WHERE NOT flagged;
```

Combine vector distance with relational rules (`amount > threshold`, geo mismatch).

---

### 1.10 Knowledge Graph / Entity Linking

**What:** Embed entity descriptions; link mentions to canonical entities by nearest neighbor.

```sql
CREATE TABLE entities (
  entity_id bigint PRIMARY KEY,
  canonical_name text,
  description text,
  embedding vector(768)
);
```

---

### 1.11 Code Search

**What:** Embed code snippets (CodeBERT, StarCoder embeddings); semantic code search in repos.

```sql
CREATE TABLE code_snippets (
  id bigserial PRIMARY KEY,
  repo text,
  path text,
  language text,
  content text,
  embedding vector(768)
);

CREATE INDEX code_lang ON code_snippets (language);
CREATE INDEX code_vec ON code_snippets USING hnsw (embedding vector_cosine_ops);
```

Filter by `language` + vector search = iterative scan or partial index per language.

---

### 1.12 Chatbot Memory & Long-Term Context

**What:** Store conversation turns or summarized memories as vectors; retrieve relevant past context.

```sql
CREATE TABLE conversation_memory (
  id bigserial PRIMARY KEY,
  user_id bigint,
  summary text,
  embedding vector(1536),
  expires_at timestamptz
);

CREATE INDEX memory_user_time ON conversation_memory (user_id, created_at DESC);
CREATE INDEX memory_vec ON conversation_memory USING hnsw (embedding vector_cosine_ops);
```

---

### 1.13 Multi-Tenant SaaS

**What:** Isolate tenants via `tenant_id` column + RLS; optional per-tenant partial indexes.

```sql
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_docs ON documents
  USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- Partial index per large tenant (optional)
CREATE INDEX docs_tenant_a ON documents USING hnsw (embedding vector_cosine_ops)
  WHERE tenant_id = 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa';
```

---

### 1.14 Molecular / Chemical Similarity (Fingerprints)

**What:** Morgan fingerprints as bit vectors; Tanimoto/Jaccard similarity.

```sql
CREATE TABLE compounds (
  id serial PRIMARY KEY,
  smiles text,
  morgan_fp bit(2048)
);

CREATE INDEX compounds_jaccard ON compounds
  USING hnsw (morgan_fp bit_jaccard_ops);
```

---

### 1.15 Geospatial + Semantic (Location-Aware Search)

**What:** Combine PostGIS distance with vector similarity (two-stage or weighted score).

```sql
SELECT id, name,
  embedding <=> $1 AS semantic_dist,
  location <-> $2::geography AS geo_dist
FROM places
WHERE ST_DWithin(location, $2::geography, 5000)  -- 5 km
ORDER BY embedding <=> $1
LIMIT 10;
```

Indexes: PostGIS GIST on `location` + HNSW on `embedding`.

---

### 1.16 Real-Time Personalization Feeds

**What:** User interest vector updated on each click; feed items ranked by similarity.

- Low latency → HNSW with tuned `ef_search`
- High write rate → consider `halfvec` or batch index rebuilds
- Cache hot user vectors in Redis; pgvector for full catalog

---

### 1.17 Compliance & eDiscovery

**What:** Find semantically similar legal documents; audit trail via Postgres WAL.

Benefit: **ACID**, point-in-time recovery, existing compliance tooling on Postgres.

---

### 1.18 IoT / Sensor Pattern Matching

**What:** Time-window embeddings of sensor readings; find similar failure patterns.

```sql
CREATE TABLE sensor_windows (
  device_id text,
  window_start timestamptz,
  pattern_embedding vector(64)
);
```

---

### Use Case → Index Quick Map

| Use Case | Typical Index | Distance |
|----------|---------------|----------|
| RAG | HNSW | Cosine |
| Product search | HNSW | Cosine |
| Recommendations | HNSW | Inner product |
| Image similarity | HNSW | Cosine or Hamming (hash) |
| Deduplication | HNSW + re-rank | Cosine / L2 |
| Fraud detection | HNSW (partial) | L2 |
| Code search | HNSW + B-tree filter | Cosine |
| Chemical similarity | HNSW | Jaccard |
| Small dataset (<10k) | None (exact) | Any |
| Huge scale, memory-bound | IVFFlat or binary quant | Cosine |

---

## 2. Index Types Overview

pgvector supports **three conceptual layers** of indexing:

```
┌─────────────────────────────────────────────────────────────────┐
│                    QUERY: ORDER BY dist LIMIT k                  │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐
   │   EXACT     │    │    HNSW     │    │    IVFFlat      │
   │  (no index) │    │  (graph)    │    │  (inverted      │
   │  seq scan   │    │             │    │   file + flat)  │
   └─────────────┘    └─────────────┘    └─────────────────┘
     100% recall        ~95-99%           ~90-98%
     O(n) scan          O(log n) approx     O(lists) approx
```

### Index access methods in pgvector

| Access Method | PostgreSQL `USING` | Purpose |
|---------------|-------------------|---------|
| Sequential scan | (none) | Exact k-NN; small tables |
| HNSW | `hnsw` | Approximate NN; best query perf |
| IVFFlat | `ivfflat` | Approximate NN; faster build, less RAM |

pgvector does **not** implement its own B-tree/GIN — you use **native Postgres indexes** alongside vector indexes for metadata filters.

### What is NOT a pgvector index

| Index | Used for |
|-------|----------|
| B-tree | `WHERE id =`, `tenant_id`, timestamps |
| GIN | Full-text, jsonb `@>` |
| GiST | PostGIS, ranges |
| BRIN | Time-series append-only |
| Hash | Equality only (rare) |

---

## 3. Exact Search (No Vector Index)

**How it works:** Postgres scans all rows (or uses parallel seq scan), computes distance for each, sorts, returns top-k.

```sql
SELECT * FROM items ORDER BY embedding <=> '[...]'::vector LIMIT 10;
```

### When exact search is correct

| Condition | Reason |
|-----------|--------|
| Table < ~10,000–50,000 rows | Seq scan + sort is fast enough |
| You need **100% recall** | Legal, safety, financial |
| Vectors change constantly | Avoid index rebuild overhead |
| Debugging / recall baseline | Compare against ANN |

### Speed up exact search

```sql
SET max_parallel_workers_per_gather = 4;

-- Normalized vectors: inner product is faster than cosine
SELECT * FROM items ORDER BY embedding <#> '[...]'::vector LIMIT 10;
```

### Exact + filter

```sql
SELECT * FROM items
WHERE category_id = 5 AND embedding IS NOT NULL
ORDER BY embedding <=> '[...]'::vector LIMIT 10;
```

B-tree on `category_id` helps the filter; vector part is still exact within filtered set.

---

## 4. HNSW Index — Full Reference

**Hierarchical Navigable Small World** — multi-layer graph. Industry standard for ANN (used in Faiss, many vector DBs).

### 4.1 How HNSW works (conceptual)

```
Layer 2 (sparse):    A ───────────── B
                      \           /
Layer 1:              A ─ C ─ D ─ B
                        \ | / \
Layer 0 (dense):        all nodes connected to neighbors
```

1. **Insert:** Assign random top layer; greedily connect to nearest neighbors per layer.
2. **Search:** Start at top layer entry point; greedy descent; refine at layer 0.
3. **ef_search:** Width of candidate list during search — higher = more nodes explored.

### 4.2 Creating HNSW indexes

```sql
-- Minimal
CREATE INDEX ON items USING hnsw (embedding vector_l2_ops);

-- Full options
CREATE INDEX items_embedding_hnsw ON items
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- Concurrent (production)
CREATE INDEX CONCURRENTLY items_embedding_hnsw ON items
  USING hnsw (embedding vector_cosine_ops);

-- After partial data load
CREATE INDEX CONCURRENTLY ...;
ANALYZE items;
```

### 4.3 Build parameters

| Parameter | Default | Range (typical) | Effect |
|-----------|---------|-----------------|--------|
| `m` | 16 | 4–48 | Connections per node; ↑ recall & RAM & build time |
| `ef_construction` | 64 | 32–512 | Build quality; ↑ recall & build time |

**Tuning guide:**

| Goal | Action |
|------|--------|
| Better recall | ↑ `ef_construction`, ↑ `m`, ↑ `hnsw.ef_search` |
| Faster builds | ↓ `ef_construction`, use binary quant index |
| Less memory | ↓ `m`, use `halfvec` expression index |
| Faster queries | Ensure index fits RAM; ↑ `ef_search` only if needed |

### 4.4 Query parameters

| Setting | Default | Description |
|---------|---------|-------------|
| `hnsw.ef_search` | 40 | Candidates explored at query time |
| `hnsw.iterative_scan` | off | `strict_order` / `relaxed_order` for filtered queries |
| `hnsw.max_scan_tuples` | 20000 | Max tuples visited in iterative mode |
| `hnsw.scan_mem_multiplier` | 1 | Memory cap multiplier × `work_mem` |

```sql
-- Session-level tuning
SET hnsw.ef_search = 100;

-- Per-transaction
BEGIN;
SET LOCAL hnsw.ef_search = 200;
SET LOCAL hnsw.iterative_scan = strict_order;
SELECT id, embedding <=> $1 AS d FROM items
WHERE active = true ORDER BY d LIMIT 20;
COMMIT;
```

### 4.5 HNSW dimension limits

| Type | Max indexed dimensions |
|------|------------------------|
| `vector` | 2,000 |
| `halfvec` | 4,000 |
| `bit` | 64,000 |
| `sparsevec` | 1,000 non-zero elements |

For OpenAI `text-embedding-3-large` (3072d): use `halfvec` index or dimensionality reduction.

### 4.6 HNSW build performance

```sql
SET maintenance_work_mem = '8GB';          -- graph should fit
SET max_parallel_maintenance_workers = 7;

-- Docker: docker run --shm-size=8g ...

-- Monitor
SELECT phase, round(100.0 * blocks_done / nullif(blocks_total, 0), 1) AS pct
FROM pg_stat_progress_create_index;
```

Phases: `initializing` → `loading tuples`

### 4.7 HNSW: inserts & updates

- **Inserts** update the graph incrementally (slower than heap insert).
- **Heavy write load:** batch inserts; consider rebuilding index off-peak.
- **VACUUM:** HNSW indexes slow vacuum; `REINDEX CONCURRENTLY` first helps.

### 4.8 HNSW: what breaks index usage

```sql
-- ✗ Won't use HNSW
ORDER BY 1 - (embedding <=> '[...]'::vector) DESC LIMIT 5;
ORDER BY l2_distance(embedding, '[...]'::vector) LIMIT 5;  -- function, not operator
SELECT * FROM items WHERE embedding <=> '[...]' < 0.5;       -- no ORDER BY + LIMIT

-- ✓ Uses HNSW
ORDER BY embedding <=> '[...]'::vector LIMIT 5;
```

---

## 5. IVFFlat Index — Full Reference

**Inverted File + Flat** — k-means clusters vectors into `lists` buckets; at query time probes nearest lists.

### 5.1 How IVFFlat works

```
Training: k-means → N lists (centroids)
Insert:   each vector assigned to nearest list
Query:    find closest probes lists → scan vectors in those lists only
```

### 5.2 Creating IVFFlat indexes

```sql
CREATE INDEX items_ivfflat ON items
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);
```

**Critical:** Train on representative data — create index **after** loading substantial rows.

### 5.3 Choosing `lists`

| Row count | Suggested `lists` |
|-----------|-------------------|
| 10,000 | 10–100 |
| 100,000 | 100–316 |
| 1,000,000 | 1,000 |
| 10,000,000 | 3,162 (`sqrt(10M)`) |

Rule of thumb:
- Up to 1M rows: `lists = rows / 1000`
- Over 1M: `lists = sqrt(rows)`

### 5.4 Query parameters

| Setting | Default | Description |
|---------|---------|-------------|
| `ivfflat.probes` | 1 | Number of lists to search |
| `ivfflat.iterative_scan` | off | `relaxed_order` for filtered queries |
| `ivfflat.max_probes` | — | Cap probes in iterative mode |

```sql
SET ivfflat.probes = 10;   -- start with sqrt(lists)

-- probes = lists → near-exact (index rarely used)
```

### 5.5 IVFFlat build phases

```sql
SELECT phase, round(100.0 * tuples_done / nullif(tuples_total, 0), 1) AS pct
FROM pg_stat_progress_create_index;
```

1. `initializing`
2. `performing k-means`
3. `assigning tuples`
4. `loading tuples` (only phase with % progress)

### 5.6 IVFFlat vs empty/small tables

If built on too little data → **terrible recall**. Drop and rebuild:

```sql
DROP INDEX items_ivfflat;
-- load more data
CREATE INDEX items_ivfflat ON items USING ivfflat (embedding vector_cosine_ops) WITH (lists = 500);
```

### 5.7 When to prefer IVFFlat

| Scenario | Why IVFFlat |
|----------|-------------|
| Very large dataset, tight memory | Smaller index than HNSW |
| Batch analytics, offline build | Faster index creation |
| Rare queries, bulk load pattern | Accept lower QPS |
| Already familiar with Faiss IVF | Similar mental model |

---

## 6. Operator Classes (Opclasses)

Each distance metric requires a matching **opclass** on the index. Mismatch = index not used.

### 6.1 `vector` opclasses

| Opclass | Operator | Distance | Typical use |
|---------|----------|----------|-------------|
| `vector_l2_ops` | `<->` | Euclidean (L2) | General embeddings |
| `vector_ip_ops` | `<#>` | Negative inner product | Normalized vectors (OpenAI) |
| `vector_cosine_ops` | `<=>` | Cosine | Text semantics |
| `vector_l1_ops` | `<+>` | Manhattan (L1) | Sparse / robust distance |

```sql
CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops);
-- Query MUST use <=>
SELECT * FROM items ORDER BY embedding <=> $1 LIMIT 10;
```

### 6.2 `halfvec` opclasses

| Opclass | Operator |
|---------|----------|
| `halfvec_l2_ops` | `<->` |
| `halfvec_ip_ops` | `<#>` |
| `halfvec_cosine_ops` | `<=>` |
| `halfvec_l1_ops` | `<+>` |

```sql
CREATE INDEX ON items USING hnsw (
  (embedding::halfvec(1536)) halfvec_cosine_ops
);
```

### 6.3 `bit` opclasses

| Opclass | Operator | Distance |
|---------|----------|----------|
| `bit_hamming_ops` | `<~>` | Hamming |
| `bit_jaccard_ops` | `<%>` | Jaccard |

### 6.4 `sparsevec` opclasses

| Opclass | Operator |
|---------|----------|
| `sparsevec_l2_ops` | `<->` |
| `sparsevec_ip_ops` | `<#>` |
| `sparsevec_cosine_ops` | `<=>` |
| `sparsevec_l1_ops` | `<+>` |

### 6.5 Multiple indexes on same column

You may need **separate indexes** per distance metric:

```sql
CREATE INDEX items_l2 ON items USING hnsw (embedding vector_l2_ops);
CREATE INDEX items_cosine ON items USING hnsw (embedding vector_cosine_ops);
```

Only one is used per query — pick the metric at index creation time.

---

## 7. Expression & Specialized Indexes

### 7.1 Half-precision expression index

Store full `vector(1536)`; index at half precision:

```sql
CREATE INDEX ON items USING hnsw (
  (embedding::halfvec(1536)) halfvec_cosine_ops
);

SELECT * FROM items
ORDER BY embedding::halfvec(1536) <=> $1::halfvec
LIMIT 10;
```

**Saves ~50% index size** with minor recall impact.

### 7.2 Binary quantization index

```sql
CREATE INDEX ON items USING hnsw (
  (binary_quantize(embedding)::bit(1536)) bit_hamming_ops
);

-- Two-stage retrieval
SELECT * FROM (
  SELECT * FROM items
  ORDER BY binary_quantize(embedding)::bit(1536) <~> binary_quantize($1::vector)
  LIMIT 50
) q ORDER BY embedding <=> $1::vector LIMIT 10;
```

| Stage | Index type | Role |
|-------|------------|------|
| 1 | Binary HNSW | Fast coarse filter |
| 2 | Exact on candidates | High recall re-rank |

### 7.3 Subvector index

Index first N dimensions (Matryoshka / MRL models):

```sql
CREATE INDEX ON items USING hnsw (
  (subvector(embedding, 1, 256)::vector(256)) vector_cosine_ops
);
```

### 7.4 Partial indexes

```sql
-- Only active products
CREATE INDEX ON products USING hnsw (embedding vector_cosine_ops)
  WHERE deleted_at IS NULL;

-- Per-category (high-traffic categories)
CREATE INDEX ON products USING hnsw (embedding vector_cosine_ops)
  WHERE category_id = 42;
```

Query must include matching `WHERE` for partial index use.

### 7.5 Multi-column / composite (relational only)

Postgres does not support multicolumn vector HNSW with B-tree in one index. Use:

1. B-tree on filter column + HNSW on vector + iterative scan
2. Partial HNSW per common filter value
3. Partition table by filter key

```sql
CREATE INDEX ON orders (customer_id);
CREATE INDEX ON orders USING hnsw (embedding vector_l2_ops);
```

### 7.6 Index on variable-dimension column

```sql
CREATE TABLE embeddings (
  model_id int,
  item_id bigint,
  embedding vector,
  PRIMARY KEY (model_id, item_id)
);

CREATE INDEX ON embeddings USING hnsw (
  (embedding::vector(1536)) vector_cosine_ops
) WHERE model_id = 1;
```

---

## 8. Combining Vector Indexes with Relational Indexes

### Pattern A: Filter-first (selective)

```sql
CREATE INDEX ON events (user_id, created_at DESC);
-- User has few rows → exact vector scan on subset is fine
SELECT * FROM events
WHERE user_id = $1
ORDER BY embedding <=> $2 LIMIT 10;
```

### Pattern B: Vector-first (ANN + post-filter)

```sql
CREATE INDEX ON catalog USING hnsw (embedding vector_cosine_ops);
SET hnsw.iterative_scan = strict_order;
SELECT * FROM catalog
WHERE in_stock = true
ORDER BY embedding <=> $1 LIMIT 20;
```

### Pattern C: Partition + per-partition HNSW

```sql
CREATE TABLE docs (
  tenant_id int,
  embedding vector(768)
) PARTITION BY LIST (tenant_id);

CREATE TABLE docs_t1 PARTITION OF docs FOR VALUES IN (1);
CREATE INDEX ON docs_t1 USING hnsw (embedding vector_cosine_ops);
```

### Pattern D: Hybrid GIN + HNSW

```sql
CREATE INDEX docs_fts ON docs USING GIN (textsearch);
CREATE INDEX docs_vec ON docs USING hnsw (embedding vector_cosine_ops);
-- Two queries in app, fuse with RRF
```

### Index inventory template

```sql
SELECT
  indexname,
  indexdef
FROM pg_indexes
WHERE tablename = 'documents';
```

---

## 9. Index Selection Decision Guide

```
START
  │
  ├─ Rows < 50k AND latency OK?
  │     └─ YES → No vector index (exact search)
  │
  ├─ Need 100% recall?
  │     └─ YES → No vector index OR ivfflat.probes = lists
  │
  ├─ Heavy filtering (>10% selectivity)?
  │     └─ YES → HNSW + iterative_scan + B-tree on filter cols
  │
  ├─ Memory constrained?
  │     ├─ YES → IVFFlat OR halfvec HNSW OR binary quant
  │     └─ NO  → HNSW
  │
  ├─ Mostly read, rare writes?
  │     └─ HNSW with high ef_construction
  │
  ├─ Heavy insert rate?
  │     └─ Batch writes; IVFFlat or rebuild HNSW periodically
  │
  └─ Dimensions > 2000?
        ├─ ≤4000 → halfvec HNSW
        ├─ ≤64000 → bit binary_quantize HNSW + re-rank
        └─ else → dimensionality reduction or subvector index
```

### HNSW vs IVFFlat summary

| Criterion | Winner |
|-----------|--------|
| Query latency | HNSW |
| Index build time | IVFFlat |
| Memory usage | IVFFlat |
| Recall @ same effort | HNSW |
| Empty table index | HNSW only |
| Write-heavy | IVFFlat (slightly) |
| Production default | **HNSW** |

---

## 10. Distance Metrics by Use Case

| Use Case | Metric | Operator | Opclass | Notes |
|----------|--------|----------|---------|-------|
| OpenAI embeddings | Inner product | `<#>` | `vector_ip_ops` | Vectors are normalized |
| Sentence-BERT | Cosine | `<=>` | `vector_cosine_ops` | Or IP if normalized |
| Image CLIP | Cosine | `<=>` | `vector_cosine_ops` | |
| User-item CF | Inner product | `<#>` | `vector_ip_ops` | After normalization |
| Geographic features | L2 | `<->` | `vector_l2_ops` | |
| Robust to outliers | L1 | `<+>` | `vector_l1_ops` | |
| Perceptual hash | Hamming | `<~>` | `bit_hamming_ops` | |
| Molecular fingerprint | Jaccard | `<%>` | `bit_jaccard_ops` | |
| Duplicate detection | L2 or cosine | `<->` / `<=>` | Match choice | Set ε threshold |

### Converting between similarity and distance

```sql
-- Cosine similarity from cosine distance
SELECT 1 - (embedding <=> $1) AS cosine_similarity;

-- Inner product (remember <#> is negative)
SELECT (embedding <#> $1) * -1 AS inner_product;
```

---

## 11. Vector Data Types by Use Case

| Type | Storage | Max dims | Index max | Best for |
|------|---------|----------|-----------|----------|
| `vector` | 4 bytes/dim + 8 | 16,000 | 2,000 | Default embeddings |
| `halfvec` | 2 bytes/dim + 8 | 16,000 | 4,000 | Memory savings |
| `bit` | 1 bit/dim + 8 | huge | 64,000 | Hashes, quantization |
| `sparsevec` | 8 bytes/nonzero + 16 | 16k nonzero | 1k nonzero | SPLADE, sparse retrieval |
| `double precision[]` | 8 bytes/dim | unlimited | via cast | High precision storage |

### Sparse vectors (learned sparse retrieval)

```sql
CREATE TABLE sparse_docs (
  id serial PRIMARY KEY,
  embedding sparsevec(30522)
);
INSERT INTO sparse_docs (embedding) VALUES ('{100:0.5,200:0.3,500:0.8}/30522');

CREATE INDEX ON sparse_docs USING hnsw (embedding sparsevec_cosine_ops);
```

---

## 12. Scale & Capacity Planning

### Index size estimation (rough)

| Index type | Rule of thumb |
|------------|---------------|
| HNSW `vector` | ~ `rows × dims × 4 × 1.5` bytes (graph overhead) |
| HNSW `halfvec` | ~ half of above |
| IVFFlat | ~ `rows × dims × 4` + list overhead |

```sql
SELECT pg_size_pretty(pg_relation_size('items_embedding_hnsw'));
SELECT pg_size_pretty(pg_total_relation_size('items'));
```

### Row counts vs strategy

| Scale | Strategy |
|-------|----------|
| < 50k | Exact search |
| 50k – 5M | HNSW, `ef_search` 40–100 |
| 5M – 50M | HNSW + halfvec or binary quant; partitioning |
| 50M+ | Citus/sharding; read replicas; IVFFlat per shard |

### RAM guideline

- Aim for **HNSW index in shared_buffers + OS cache**
- If index >> RAM → IVFFlat, binary quant, or more nodes

### Connection pooling

- PgBouncer transaction mode for short vector queries
- Separate read replicas for search-heavy workloads

---

## 13. pgvector vs Dedicated Vector Databases

| Factor | pgvector | Pinecone / Weaviate / Qdrant |
|--------|----------|------------------------------|
| Ops model | Same as Postgres | New infrastructure |
| JOINs with relational data | Native | Requires sync / dual writes |
| ACID transactions | Yes | Varies |
| Hybrid SQL + vectors | Native | Limited |
| Managed ANN tuning | Manual (HNSW params) | Often auto-tuned |
| Max scale (single node) | Millions | Billions (distributed) |
| Replication | Postgres streaming | Built-in |
| Team skill | SQL/Postgres | New API surface |

**Choose pgvector when:** data already in Postgres, need JOINs/RLS/transactions, moderate scale, unified ops.

**Choose dedicated vector DB when:** billions of vectors, sub-ms at huge scale, specialized multi-tenancy, minimal relational needs.

---

## 14. Anti-Patterns & Common Mistakes

| Mistake | Why it's bad | Fix |
|---------|--------------|-----|
| Index before bulk load | Slow load; suboptimal IVFFlat | Load first, index after |
| Wrong opclass for operator | Index ignored | Match `vector_cosine_ops` with `<=>` |
| `ORDER BY similarity DESC` | Expression blocks index | `ORDER BY embedding <=> $q LIMIT k` |
| One index for all tenants unfiltered | Poor recall with filters | Iterative scan or partial indexes |
| `lists = 10` on 10M rows | Awful IVFFlat recall | `sqrt(rows)` |
| Ignoring NULL embeddings | Surprising missing results | `WHERE embedding IS NOT NULL` |
| Zero vector + cosine | Not indexed | Validate inputs |
| No `ANALYZE` after bulk load | Bad planner choices | `ANALYZE table` |
| `ef_search = 5` | Very low recall | Start at 40–100 |
| Storing wrong dimension | Insert errors | Match model output to `vector(n)` |
| Multiple PG versions installed | Extension mismatch | Align package with running server |

### Recall validation script

```sql
-- Compare top-10: exact vs approximate
WITH exact AS (
  SELECT id FROM items
  ORDER BY embedding <=> $1
  LIMIT 10
),
approx AS (
  SELECT id FROM items
  ORDER BY embedding <=> $1  -- uses HNSW if enabled
  LIMIT 10
)
SELECT
  (SELECT count(*) FROM exact e JOIN approx a ON e.id = a.id) AS overlap,
  10 AS k,
  round(100.0 * (SELECT count(*) FROM exact e JOIN approx a ON e.id = a.id) / 10, 1) AS recall_pct;
```

---

## Appendix: Complete index DDL cheat sheet

```sql
-- HNSW cosine (RAG default)
CREATE INDEX CONCURRENTLY idx_docs_vec ON documents
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- HNSW inner product (normalized OpenAI)
CREATE INDEX CONCURRENTLY idx_docs_ip ON documents
  USING hnsw (embedding vector_ip_ops);

-- IVFFlat (memory-conscious)
CREATE INDEX CONCURRENTLY idx_docs_ivf ON documents
  USING ivfflat (embedding vector_cosine_ops) WITH (lists = 1000);

-- Half-precision index
CREATE INDEX CONCURRENTLY idx_docs_half ON documents
  USING hnsw ((embedding::halfvec(1536)) halfvec_cosine_ops);

-- Binary quantization
CREATE INDEX CONCURRENTLY idx_docs_bin ON documents
  USING hnsw ((binary_quantize(embedding)::bit(1536)) bit_hamming_ops);

-- Partial (active rows only)
CREATE INDEX CONCURRENTLY idx_docs_active ON documents
  USING hnsw (embedding vector_cosine_ops) WHERE status = 'active';

-- Full-text companion
CREATE INDEX idx_docs_fts ON documents USING GIN (textsearch);

-- Metadata filter
CREATE INDEX idx_docs_tenant ON documents (tenant_id);
CREATE INDEX idx_docs_meta ON documents USING GIN (metadata);
```

---

*Document version: pgvector 0.8.x · PostgreSQL 13–18*
