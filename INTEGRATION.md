# pgvector Application Integration Guide

How to use pgvector from application code — covering client libraries, ORMs, embedding pipelines, and production patterns.

---

## Table of Contents

1. [Integration Overview](#1-integration-overview)
2. [Connection & Type Registration](#2-connection--type-registration)
3. [Python](#3-python)
4. [Node.js / TypeScript](#4-nodejs--typescript)
5. [Go](#5-go)
6. [Java / Kotlin](#6-java--kotlin)
7. [Rust](#7-rust)
8. [Ruby](#8-ruby)
9. [Other Languages](#9-other-languages)
10. [RAG Pipeline Pattern](#10-rag-pipeline-pattern)
11. [Embedding Providers](#11-embedding-providers)
12. [Production Patterns](#12-production-patterns)

---

## 1. Integration Overview

pgvector is a **PostgreSQL extension**. Application integration involves:

1. **Enable extension** in your database (`CREATE EXTENSION vector`)
2. **Register vector types** with your DB driver (most client libraries need this)
3. **Store embeddings** as `vector(n)` columns
4. **Query** with distance operators (`<->`, `<=>`, `<#>`) and `ORDER BY ... LIMIT k`

### Client library ecosystem

| Language | Package | Frameworks |
|----------|---------|------------|
| Python | [`pgvector`](https://github.com/pgvector/pgvector-python) | Django, SQLAlchemy, Psycopg, asyncpg |
| Node.js | [`pgvector`](https://github.com/pgvector/pgvector-node) | Prisma, Drizzle, TypeORM, Knex, Kysely |
| Go | [`pgvector-go`](https://github.com/pgvector/pgvector-go) | pgx, GORM, Ent, Bun |
| Java | [`pgvector-java`](https://github.com/pgvector/pgvector-java) | JDBC, Spring |
| Rust | [`pgvector-rust`](https://github.com/pgvector/pgvector-rust) | sqlx, tokio-postgres |
| Ruby | [`neighbor`](https://github.com/ankane/neighbor) | ActiveRecord |
| .NET | [`pgvector-dotnet`](https://github.com/pgvector/pgvector-dotnet) | Npgsql, EF Core |

---

## 2. Connection & Type Registration

Most drivers cannot serialize `float[]` → `vector` without registration.

### Connection string

```
postgresql://user:password@localhost:5432/vectordb?sslmode=prefer
```

### Environment variables (typical)

```bash
DATABASE_URL=postgresql://app:secret@localhost:5432/vectordb
OPENAI_API_KEY=sk-...
```

### Startup checklist

```sql
-- Run on app bootstrap or in migrations
CREATE EXTENSION IF NOT EXISTS vector;
```

---

## 3. Python

### Install

```bash
pip install pgvector psycopg[binary]   # Psycopg 3
# or
pip install pgvector asyncpg           # async
# or
pip install pgvector sqlalchemy        # ORM
```

### 3.1 Psycopg 3 (recommended)

```python
import psycopg
import numpy as np
from pgvector.psycopg import register_vector

conn = psycopg.connect("postgresql://localhost/vectordb")
register_vector(conn)

with conn.cursor() as cur:
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id bigserial PRIMARY KEY,
            content text,
            embedding vector(1536)
        )
    """)

    # Insert
    embedding = np.random.rand(1536).astype(np.float32)
    cur.execute(
        "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
        ("Hello world", embedding)
    )

    # k-NN search (cosine)
    cur.execute("""
        SELECT id, content, embedding <=> %s AS distance
        FROM documents
        ORDER BY embedding <=> %s
        LIMIT 5
    """, (embedding, embedding))
    results = cur.fetchall()

conn.commit()
```

**Connection pool:**

```python
from psycopg_pool import ConnectionPool

def configure(conn):
    register_vector(conn)

pool = ConnectionPool("postgresql://localhost/vectordb", configure=configure)
```

**Async:**

```python
import psycopg
from pgvector.psycopg import register_vector_async

async with await psycopg.AsyncConnection.connect(dsn) as conn:
    await register_vector_async(conn)
    async with conn.cursor() as cur:
        await cur.execute("SELECT * FROM documents ORDER BY embedding <=> %s LIMIT 5", (embedding,))
```

### 3.2 SQLAlchemy 2.0

```python
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import VECTOR
from pgvector.psycopg import register_vector
from sqlalchemy import event

engine = create_engine("postgresql+psycopg://localhost/vectordb")

@event.listens_for(engine, "connect")
def connect(dbapi_connection, connection_record):
    register_vector(dbapi_connection)

class Base(DeclarativeBase):
    pass

class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(primary_key=True)
    content: Mapped[str]
    embedding: Mapped[list] = mapped_column(VECTOR(1536))

with engine.begin() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(conn)

# Nearest neighbors
with Session(engine) as session:
    results = session.scalars(
        select(Document)
        .order_by(Document.embedding.cosine_distance(query_embedding))
        .limit(5)
    ).all()
```

**HNSW index via SQLAlchemy:**

```python
from sqlalchemy import Index

Index(
    "documents_embedding_idx",
    Document.embedding,
    postgresql_using="hnsw",
    postgresql_with={"m": 16, "ef_construction": 64},
    postgresql_ops={"embedding": "vector_cosine_ops"},
).create(engine)
```

### 3.3 Django

```python
# models.py
from django.db import models
from pgvector.django import VectorField, HnswIndex, CosineDistance

class Document(models.Model):
    content = models.TextField()
    embedding = VectorField(dimensions=1536)

    class Meta:
        indexes = [
            HnswIndex(
                name="documents_embedding_idx",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_cosine_ops"],
            )
        ]

# views.py
def search(query_embedding):
    return Document.objects.order_by(
        CosineDistance("embedding", query_embedding)
    )[:5]
```

**Migration to enable extension:**

```python
from pgvector.django import VectorExtension

class Migration(migrations.Migration):
    operations = [VectorExtension()]
```

### 3.4 FastAPI + asyncpg

```python
import asyncpg
from pgvector.asyncpg import register_vector
from fastapi import FastAPI

app = FastAPI()
pool = None

async def init_db():
    global pool
    async def init(conn):
        await register_vector(conn)
    pool = await asyncpg.create_pool(dsn, init=init)
    async with pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

@app.on_event("startup")
async def startup():
    await init_db()

@app.get("/search")
async def search(q: str):
    embedding = await get_embedding(q)  # your embedding function
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, content, embedding <=> $1 AS distance
            FROM documents ORDER BY embedding <=> $1 LIMIT 5
        """, embedding)
    return [dict(r) for r in rows]
```

### 3.5 LangChain integration

```python
from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings

CONNECTION_STRING = "postgresql+psycopg://user:pass@localhost/vectordb"

vectorstore = PGVector(
    embeddings=OpenAIEmbeddings(),
    collection_name="documents",
    connection_string=CONNECTION_STRING,
    use_jsonb=True,
)

# Add documents
vectorstore.add_texts(["Hello world", "PostgreSQL vectors"])

# Search
docs = vectorstore.similarity_search("hello", k=5)
```

### 3.6 LlamaIndex integration

```python
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

vector_store = PGVectorStore.from_params(
    database="vectordb",
    host="localhost",
    password="secret",
    port=5432,
    user="app",
    table_name="llama_documents",
    embed_dim=1536,
)

index = VectorStoreIndex.from_documents(
    SimpleDirectoryReader("./data").load_data(),
    vector_store=vector_store,
)
query_engine = index.as_query_engine()
response = query_engine.query("What is pgvector?")
```

---

## 4. Node.js / TypeScript

### Install

```bash
npm install pgvector pg
# or for Prisma/Drizzle, see below
```

### 4.1 node-postgres (pg)

```javascript
import pg from 'pg';
import pgvector from 'pgvector/pg';

const pool = new pg.Pool({
  connectionString: process.env.DATABASE_URL,
  onConnect: async (client) => await pgvector.registerTypes(client),
});

const client = await pool.connect();
await client.query('CREATE EXTENSION IF NOT EXISTS vector');

const embedding = [0.1, 0.2, 0.3]; // your 1536-dim array

await client.query(
  'INSERT INTO documents (content, embedding) VALUES ($1, $2)',
  ['Hello', pgvector.toSql(embedding)]
);

const { rows } = await client.query(
  'SELECT id, content FROM documents ORDER BY embedding <=> $1 LIMIT 5',
  [pgvector.toSql(embedding)]
);
```

### 4.2 Drizzle ORM

```typescript
import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import { pgTable, serial, text, vector } from 'drizzle-orm/pg-core';
import { cosineDistance } from 'drizzle-orm';

const client = postgres(process.env.DATABASE_URL!);
const db = drizzle(client);

const documents = pgTable('documents', {
  id: serial('id').primaryKey(),
  content: text('content').notNull(),
  embedding: vector('embedding', { dimensions: 1536 }),
});

await client`CREATE EXTENSION IF NOT EXISTS vector`;

await db.insert(documents).values({
  content: 'Hello',
  embedding: embeddingArray,
});

const results = await db
  .select()
  .from(documents)
  .orderBy(cosineDistance(documents.embedding, embeddingArray))
  .limit(5);
```

### 4.3 Prisma

```prisma
// schema.prisma
generator client {
  provider        = "prisma-client-js"
  previewFeatures = ["postgresqlExtensions"]
}

datasource db {
  provider   = "postgresql"
  url        = env("DATABASE_URL")
  extensions = [vector]
}

model Document {
  id        Int                       @id @default(autoincrement())
  content   String
  embedding Unsupported("vector(1536)")?
}
```

```typescript
import { PrismaClient } from '@prisma/client';
import pgvector from 'pgvector';

const prisma = new PrismaClient();

const embedding = pgvector.toSql(queryEmbedding);

const docs = await prisma.$queryRaw`
  SELECT id, content, embedding <=> ${embedding}::vector AS distance
  FROM "Document"
  ORDER BY embedding <=> ${embedding}::vector
  LIMIT 5
`;
```

> Prisma Migrate does not support HNSW indexes — create them in a raw SQL migration.

### 4.4 TypeORM

```typescript
import 'reflect-metadata';
import { DataSource, Entity, PrimaryGeneratedColumn, Column } from 'typeorm';
import pgvector from 'pgvector';

@Entity()
class Document {
  @PrimaryGeneratedColumn()
  id: number;

  @Column('text')
  content: string;

  @Column('vector', { length: 1536 })
  embedding: number[];
}

const AppDataSource = new DataSource({
  type: 'postgres',
  url: process.env.DATABASE_URL,
  entities: [Document],
});

await AppDataSource.query('CREATE EXTENSION IF NOT EXISTS vector');

const items = await AppDataSource.getRepository(Document)
  .createQueryBuilder('doc')
  .orderBy('doc.embedding <=> :embedding')
  .setParameters({ embedding: pgvector.toSql(queryEmbedding) })
  .limit(5)
  .getMany();
```

### 4.5 Express API example

```javascript
import express from 'express';
import pg from 'pg';
import pgvector from 'pgvector/pg';
import OpenAI from 'openai';

const app = express();
const pool = new pg.Pool({
  connectionString: process.env.DATABASE_URL,
  onConnect: (c) => pgvector.registerTypes(c),
});
const openai = new OpenAI();

app.post('/search', async (req, res) => {
  const { query } = req.body;
  const emb = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: query,
  });
  const vector = emb.data[0].embedding;

  const { rows } = await pool.query(
    `SELECT id, content, 1 - (embedding <=> $1) AS score
     FROM documents ORDER BY embedding <=> $1 LIMIT 10`,
    [pgvector.toSql(vector)]
  );
  res.json(rows);
});

app.listen(3000);
```

---

## 5. Go

### Install

```bash
go get github.com/pgvector/pgvector-go
go get github.com/pgvector/pgvector-go/pgx
go get github.com/jackc/pgx/v5
```

### pgx example

```go
package main

import (
    "context"
    "github.com/jackc/pgx/v5"
    "github.com/jackc/pgx/v5/pgxpool"
    "github.com/pgvector/pgvector-go"
    pgxvec "github.com/pgvector/pgvector-go/pgx"
)

func main() {
    ctx := context.Background()
    config, _ := pgxpool.ParseConfig("postgresql://localhost/vectordb")
    config.AfterConnect = func(ctx context.Context, conn *pgx.Conn) error {
        return pgxvec.RegisterTypes(ctx, conn)
    }
    pool, _ := pgxpool.NewWithConfig(ctx, config)
    defer pool.Close()

    conn, _ := pool.Acquire(ctx)
    defer conn.Release()

    conn.Exec(ctx, "CREATE EXTENSION IF NOT EXISTS vector")

    vec := pgvector.NewVector([]float32{0.1, 0.2, 0.3})
    conn.Exec(ctx, "INSERT INTO documents (content, embedding) VALUES ($1, $2)",
        "Hello", vec)

    rows, _ := conn.Query(ctx,
        "SELECT id, content FROM documents ORDER BY embedding <=> $1 LIMIT 5", vec)
    defer rows.Close()
    for rows.Next() {
        var id int64
        var content string
        rows.Scan(&id, &content)
    }
}
```

### GORM example

```go
type Document struct {
    ID        uint `gorm:"primaryKey"`
    Content   string
    Embedding pgvector.Vector `gorm:"type:vector(1536)"`
}

db.Exec("CREATE EXTENSION IF NOT EXISTS vector")

item := Document{
    Content:   "Hello",
    Embedding: pgvector.NewVector(embeddingSlice),
}
db.Create(&item)

var results []Document
db.Clauses(clause.OrderBy{
    Expression: clause.Expr{
        SQL:  "embedding <=> ?",
        Vars: []interface{}{pgvector.NewVector(queryVec)},
    },
}).Limit(5).Find(&results)
```

---

## 6. Java / Kotlin

### Maven dependency

```xml
<dependency>
  <groupId>com.pgvector</groupId>
  <artifactId>pgvector</artifactId>
  <version>0.1.6</version>
</dependency>
```

### JDBC example

```java
import com.pgvector.PGvector;
import java.sql.*;

PGvector.registerTypes(connection);
connection.createStatement().execute("CREATE EXTENSION IF NOT EXISTS vector");

PGvector embedding = new PGvector(new float[]{0.1f, 0.2f, 0.3f});

PreparedStatement insert = connection.prepareStatement(
    "INSERT INTO documents (content, embedding) VALUES (?, ?)");
insert.setString(1, "Hello");
insert.setObject(2, embedding);
insert.executeUpdate();

PreparedStatement search = connection.prepareStatement("""
    SELECT id, content, embedding <=> ? AS distance
    FROM documents ORDER BY embedding <=> ? LIMIT 5
""");
search.setObject(1, embedding);
search.setObject(2, embedding);
ResultSet rs = search.executeQuery();
```

### Spring Boot

```kotlin
// build.gradle.kts
implementation("com.pgvector:pgvector:0.1.6")
implementation("org.springframework.boot:spring-boot-starter-data-jpa")
```

Use native queries for vector search; JPA does not natively understand `vector` type.

---

## 7. Rust

### Cargo.toml

```toml
[dependencies]
pgvector = "0.4"
sqlx = { version = "0.8", features = ["postgres", "runtime-tokio"] }
tokio = { version = "1", features = ["full"] }
```

### sqlx example

```rust
use pgvector::Vector;
use sqlx::postgres::PgPoolOptions;

#[tokio::main]
async fn main() -> Result<(), sqlx::Error> {
    let pool = PgPoolOptions::new()
        .connect("postgresql://localhost/vectordb")
        .await?;

    sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
        .execute(&pool).await?;

    let embedding = Vector::from(vec![0.1_f32, 0.2, 0.3]);

    sqlx::query("INSERT INTO documents (content, embedding) VALUES ($1, $2)")
        .bind("Hello")
        .bind(&embedding)
        .execute(&pool).await?;

    let rows = sqlx::query_as::<_, (i64, String)>(
        "SELECT id, content FROM documents ORDER BY embedding <=> $1 LIMIT 5"
    )
    .bind(&embedding)
    .fetch_all(&pool).await?;

    Ok(())
}
```

---

## 8. Ruby

### neighbor gem (ActiveRecord)

```ruby
# Gemfile
gem "neighbor"

# migration
class InstallNeighbor < ActiveRecord::Migration[7.1]
  def change
    enable_extension "vector"
    create_table :documents do |t|
      t.text :content
      t.vector :embedding, limit: 1536
      t.timestamps
    end
    add_index :documents, :embedding, using: :hnsw, opclass: :vector_cosine_ops
  end
end

# model
class Document < ApplicationRecord
  has_neighbors :embedding
end

# search
query_embedding = get_embedding("search query")
Document.nearest_neighbors(:embedding, query_embedding, distance: "cosine").limit(5)
```

---

## 9. Other Languages

| Language | Library |
|----------|---------|
| C# / .NET | [pgvector-dotnet](https://github.com/pgvector/pgvector-dotnet) |
| PHP | [pgvector-php](https://github.com/pgvector/pgvector-php) |
| Elixir | [pgvector-elixir](https://github.com/pgvector/pgvector-elixir) |
| Swift | [pgvector-swift](https://github.com/pgvector/pgvector-swift) |
| Dart | [pgvector-dart](https://github.com/pgvector/pgvector-dart) |

All follow the same pattern: register types → insert `vector` → query with distance operators.

---

## 10. RAG Pipeline Pattern

### Architecture

```
User Query
    ↓
[Embedding Model] → query vector
    ↓
[pgvector k-NN] → top-k document chunks
    ↓
[LLM with context] → answer
```

### Chunking strategy

```python
def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks
```

### Ingestion pipeline

```python
async def ingest_document(source: str, content: str, pool):
    chunks = chunk_text(content)
    embeddings = await embed_batch(chunks)

    async with pool.acquire() as conn:
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            await conn.execute("""
                INSERT INTO knowledge_base (source, chunk_index, content, embedding)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (source, chunk_index) DO UPDATE
                SET content = EXCLUDED.content, embedding = EXCLUDED.embedding
            """, source, i, chunk, emb)
```

### Retrieval with metadata filter

```python
async def retrieve(query: str, tenant_id: str, k: int = 5):
    embedding = await embed(query)
    async with pool.acquire() as conn:
        return await conn.fetch("""
            SELECT content, metadata, embedding <=> $1 AS distance
            FROM knowledge_base
            WHERE metadata->>'tenant_id' = $2
            ORDER BY embedding <=> $1
            LIMIT $3
        """, embedding, tenant_id, k)
```

### Hybrid search (RRF)

```python
def reciprocal_rank_fusion(rankings: list[list], k: int = 60) -> list:
    """Combine multiple ranked lists."""
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

# Get vector ranking
vector_ids = [r['id'] for r in vector_results]
# Get full-text ranking
text_ids = [r['id'] for r in fts_results]
# Fuse
fused = reciprocal_rank_fusion([vector_ids, text_ids])
```

---

## 11. Embedding Providers

### OpenAI

```python
from openai import OpenAI
client = OpenAI()

def embed(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model="text-embedding-3-small",  # 1536 dims
        input=texts,
    )
    return [d.embedding for d in response.data]
```

| Model | Dimensions | Notes |
|-------|------------|-------|
| `text-embedding-3-small` | 1536 (configurable) | Cost-effective |
| `text-embedding-3-large` | 3072 | Higher quality |
| `text-embedding-ada-002` | 1536 | Legacy |

### Local (Sentence Transformers)

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384 dims

embeddings = model.encode(["hello world", "pgvector rocks"])
```

### Ollama

```python
import ollama
response = ollama.embeddings(model="nomic-embed-text", prompt="hello")
embedding = response["embedding"]
```

### Cohere (binary embeddings)

```python
import cohere
co = cohere.Client()
response = co.embed(texts=["hello"], model="embed-english-v3.0", input_type="search_document")
```

---

## 12. Production Patterns

### Migrations

Always enable extension in migrations, not at runtime:

```sql
-- migrations/001_enable_pgvector.sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Index creation in production

```sql
-- After bulk load
CREATE INDEX CONCURRENTLY documents_embedding_hnsw_idx
  ON documents USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
```

### Connection pooling

Use PgBouncer in **transaction mode** for most apps. pgvector queries are short read transactions.

```
# pgbouncer.ini
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 20
```

### Batch embedding on ingest

```python
BATCH_SIZE = 100

for batch in chunked(documents, BATCH_SIZE):
    texts = [d.content for d in batch]
    embeddings = await embed_batch(texts)
    await bulk_insert(batch, embeddings)
```

### Dimension consistency

Validate embedding dimensions match column definition:

```python
assert len(embedding) == 1536, f"Expected 1536 dims, got {len(embedding)}"
```

### Error handling

```python
try:
    results = await search(embedding)
except asyncpg.DataError as e:
    if "different vector dimensions" in str(e):
        raise ValueError("Embedding dimension mismatch") from e
    raise
```

### Security

- Use parameterized queries (never string-interpolate vectors)
- Row-level security for multi-tenant:

```sql
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON documents
  USING (tenant_id = current_setting('app.tenant_id')::uuid);
```

### Monitoring query latency

```sql
-- Log slow vector queries
ALTER SYSTEM SET log_min_duration_statement = 1000;  -- 1s
SELECT pg_reload_conf();
```

---

## Quick Reference: Distance by Framework

| Framework | L2 | Cosine | Inner Product |
|-----------|-----|--------|---------------|
| Raw SQL | `ORDER BY col <-> $1` | `ORDER BY col <=> $1` | `ORDER BY col <#> $1` |
| SQLAlchemy | `.l2_distance(v)` | `.cosine_distance(v)` | `.max_inner_product(v)` |
| Django | `L2Distance('col', v)` | `CosineDistance('col', v)` | `MaxInnerProduct('col', v)` |
| Drizzle | `l2Distance(col, v)` | `cosineDistance(col, v)` | `innerProduct(col, v)` |
| Knex/Kysely | `l2Distance('col', v)` | `cosineDistance('col', v)` | `maxInnerProduct('col', v)` |
| Go Ent | `entvec.L2Distance(...)` | `entvec.CosineDistance(...)` | `entvec.MaxInnerProduct(...)` |

---

See [REFERENCE.md](./REFERENCE.md) for installation, SQL reference, indexing, and troubleshooting.
