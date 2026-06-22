"""Compare L2, cosine, and inner product on same vectors."""

import numpy as np

from db import EMBED_DIM, print_results, reset_schema


def run(conn) -> None:
    print("\n--- Distance Metrics Comparison ---")
    print("Demonstrates: <-> L2, <=> cosine, <#> inner product\n")

    reset_schema(conn)
    conn.execute(f"""
        CREATE TABLE points (
            id serial PRIMARY KEY,
            label text,
            embedding vector({EMBED_DIM})
        )
    """)

    # Three vectors: normalized for IP/cosine comparison
    raw = [
        ("query", np.random.RandomState(42).randn(EMBED_DIM).astype(np.float32)),
        ("close", np.random.RandomState(1).randn(EMBED_DIM).astype(np.float32)),
        ("far", np.random.RandomState(99).randn(EMBED_DIM).astype(np.float32)),
    ]
    for label, vec in raw:
        vec = vec / np.linalg.norm(vec)
        conn.execute(
            "INSERT INTO points (label, embedding) VALUES (%s, %s)",
            (label, vec),
        )

    query = conn.execute(
        "SELECT embedding FROM points WHERE label = 'query'"
    ).fetchone()[0]

    rows = conn.execute(
        """
        SELECT label,
               round((embedding <-> %s)::numeric, 4) AS l2,
               round((embedding <=> %s)::numeric, 4) AS cosine_dist,
               round(((embedding <#> %s) * -1)::numeric, 4) AS inner_product
        FROM points
        WHERE label != 'query'
        ORDER BY embedding <=> %s
        """,
        (query, query, query, query),
    ).fetchall()

    print("Distances from 'query' vector:")
    print_results(rows, ["label", "L2 (<->)", "cosine (<=>)", "inner_product (<#>)"])

    print("\nOperator guide:")
    print("  <->  L2 distance       — general purpose")
    print("  <=>  Cosine distance   — text/embeddings (1 - cosine_similarity)")
    print("  <#>  Neg inner product — use when vectors are L2-normalized (OpenAI)")
    print("  <+>  L1 distance       — robust to outliers")

    # Show index opclass must match operator
    conn.execute("""
        CREATE INDEX points_cosine ON points
        USING hnsw (embedding vector_cosine_ops)
    """)
    conn.execute("""
        CREATE INDEX points_l2 ON points
        USING hnsw (embedding vector_l2_ops)
    """)
    print("\nCreated separate HNSW indexes: vector_cosine_ops + vector_l2_ops")
    print("(Query operator must match index opclass for index to be used)")
