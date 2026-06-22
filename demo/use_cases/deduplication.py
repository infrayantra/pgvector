"""Near-duplicate detection using distance threshold."""

from db import EMBED_DIM, print_results, reset_schema
from embeddings import embed


TEXTS = [
    "PostgreSQL is a powerful open-source relational database system.",
    "PostgreSQL is a powerful open source relational database.",  # near-duplicate
    "Redis is an in-memory data structure store used as cache.",
    "MongoDB is a document-oriented NoSQL database program.",
    "Postgres is a robust open-source relational DBMS.",  # near-duplicate of first
    "Python is a high-level programming language.",
]


def run(conn) -> None:
    print("\n--- Duplicate / Near-Duplicate Detection ---")
    print("Demonstrates: pairwise distance, threshold-based dedup\n")

    reset_schema(conn)
    conn.execute(f"""
        CREATE TABLE records (
            id serial PRIMARY KEY,
            content text NOT NULL,
            embedding vector({EMBED_DIM}),
            is_duplicate bool DEFAULT false
        )
    """)
    conn.execute("""
        CREATE INDEX records_hnsw ON records
        USING hnsw (embedding vector_cosine_ops)
    """)

    vectors = embed(TEXTS)
    for text, vec in zip(TEXTS, vectors):
        conn.execute(
            "INSERT INTO records (content, embedding) VALUES (%s, %s)",
            (text, vec),
        )

    threshold = 0.15
    print(f"Similarity threshold: cosine distance < {threshold}\n")

    rows = conn.execute(
        """
        SELECT a.id, left(a.content, 50) AS content_a,
               b.id, left(b.content, 50) AS content_b,
               round((a.embedding <=> b.embedding)::numeric, 4) AS distance
        FROM records a
        JOIN records b ON a.id < b.id
        WHERE a.embedding <=> b.embedding < %s
        ORDER BY distance
        """,
        (threshold,),
    ).fetchall()

    print("Near-duplicate pairs:")
    print_results(rows, ["id_a", "content_a", "id_b", "content_b", "distance"])

    # Mark duplicates (keep lowest id)
    conn.execute(
        """
        UPDATE records SET is_duplicate = true
        WHERE id IN (
            SELECT b.id FROM records a
            JOIN records b ON a.id < b.id
            WHERE a.embedding <=> b.embedding < %s
        )
        """,
        (threshold,),
    )

    unique = conn.execute(
        "SELECT id, left(content, 55) FROM records WHERE NOT is_duplicate ORDER BY id"
    ).fetchall()
    print("\nUnique records after dedup:")
    print_results(unique, ["id", "content"])
