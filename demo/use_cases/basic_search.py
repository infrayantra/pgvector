"""Basic k-NN without approximate index (exact search)."""

import numpy as np

from db import EMBED_DIM, explain_query, print_results, reset_schema
from embeddings import embed


def run(conn) -> None:
    print("\n--- Basic Vector Search (Exact) ---")
    print("Demonstrates: CREATE TABLE, INSERT, ORDER BY <-> LIMIT (no ANN index)\n")

    reset_schema(conn)
    conn.execute(f"""
        CREATE TABLE items (
            id serial PRIMARY KEY,
            name text NOT NULL,
            embedding vector({EMBED_DIM})
        )
    """)

    items = [
        ("PostgreSQL", "Open-source relational database"),
        ("Redis", "In-memory key-value store"),
        ("pgvector", "Vector similarity search for Postgres"),
        ("Elasticsearch", "Distributed search and analytics engine"),
        ("MongoDB", "Document-oriented NoSQL database"),
    ]

    for name, desc in items:
        vec = embed([f"{name}: {desc}"])[0]
        conn.execute(
            "INSERT INTO items (name, embedding) VALUES (%s, %s)",
            (name, vec),
        )

    query_vec = embed(["vector database for AI"])[0]
    rows = conn.execute(
        """
        SELECT name, round((embedding <=> %s)::numeric, 4) AS cosine_distance
        FROM items
        ORDER BY embedding <=> %s
        LIMIT 3
        """,
        (query_vec, query_vec),
    ).fetchall()

    print("Query: 'vector database for AI'")
    print("\nTop 3 nearest (cosine distance, lower = closer):")
    print_results(rows, ["name", "cosine_distance"])

    plan = explain_query(
        conn,
        f"SELECT name FROM items ORDER BY embedding <=> %s LIMIT 3",
        query_vec,
    )
    print("\nQuery plan (note: Seq Scan = exact search):")
    for line in plan.split("\n")[:6]:
        print(f"  {line}")
