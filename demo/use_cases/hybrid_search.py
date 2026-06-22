"""Hybrid search: PostgreSQL full-text + vector with RRF fusion."""

from db import EMBED_DIM, print_results, reset_schema
from embeddings import embed


ARTICLES = [
    ("a1", "PostgreSQL Performance Tuning Guide", "Learn how to tune shared_buffers and work_mem for better query performance."),
    ("a2", "Introduction to Vector Databases", "Vector databases store embeddings for similarity search in AI applications."),
    ("a3", "Full-Text Search in Postgres", "PostgreSQL provides powerful full-text search with tsvector and GIN indexes."),
    ("a4", "Machine Learning with Python", "Train neural networks using PyTorch and scikit-learn for classification tasks."),
    ("a5", "pgvector Hybrid Search Patterns", "Combine vector similarity with keyword search using reciprocal rank fusion."),
    ("a6", "Database Indexing Strategies", "B-tree, GIN, GiST, and HNSW indexes serve different query patterns."),
]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def run(conn) -> None:
    print("\n--- Hybrid Search (Full-Text + Vector) ---")
    print("Demonstrates: tsvector GIN index + HNSW + Reciprocal Rank Fusion\n")

    reset_schema(conn)
    conn.execute(f"""
        CREATE TABLE articles (
            id text PRIMARY KEY,
            title text NOT NULL,
            body text NOT NULL,
            textsearch tsvector GENERATED ALWAYS AS (
                to_tsvector('english', title || ' ' || body)
            ) STORED,
            embedding vector({EMBED_DIM})
        )
    """)
    conn.execute("CREATE INDEX articles_fts ON articles USING GIN (textsearch)")
    conn.execute("""
        CREATE INDEX articles_hnsw ON articles
        USING hnsw (embedding vector_cosine_ops)
    """)

    for doc_id, title, body in ARTICLES:
        vec = embed([f"{title}. {body}"])[0]
        conn.execute(
            "INSERT INTO articles (id, title, body, embedding) VALUES (%s, %s, %s, %s)",
            (doc_id, title, body, vec),
        )

    query = "postgres vector search indexing"
    qvec = embed([query])[0]

    vector_rows = conn.execute(
        """
        SELECT id, title FROM articles
        ORDER BY embedding <=> %s LIMIT 5
        """,
        (qvec,),
    ).fetchall()
    vector_ids = [r[0] for r in vector_rows]

    fts_rows = conn.execute(
        """
        SELECT id, title, ts_rank_cd(textsearch, query) AS rank
        FROM articles, plainto_tsquery('english', %s) query
        WHERE textsearch @@ query
        ORDER BY rank DESC LIMIT 5
        """,
        (query,),
    ).fetchall()
    fts_ids = [r[0] for r in fts_rows]

    print(f'Query: "{query}"\n')
    print("Vector search top 5:")
    print_results(vector_rows, ["id", "title"])
    print("\nFull-text search top 5:")
    print_results([(r[0], r[1], round(r[2], 4)) for r in fts_rows], ["id", "title", "rank"])

    fused = reciprocal_rank_fusion([vector_ids, fts_ids])
    print("\nReciprocal Rank Fusion (combined):")
    id_to_title = {a[0]: a[1] for a in ARTICLES}
    fused_rows = [(doc_id, id_to_title[doc_id], round(score, 4)) for doc_id, score in fused[:5]]
    print_results(fused_rows, ["id", "title", "rrf_score"])
