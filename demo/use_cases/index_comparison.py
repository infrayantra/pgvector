"""Compare exact sequential scan vs HNSW approximate index."""

import time

import numpy as np

from db import EMBED_DIM, print_results, reset_schema


def _seed_data(conn, n: int = 2000) -> np.ndarray:
    rng = np.random.RandomState(0)
    query = rng.randn(EMBED_DIM).astype(np.float32)
    query /= np.linalg.norm(query)

    conn.execute(f"""
        CREATE TABLE vectors (
            id serial PRIMARY KEY,
            embedding vector({EMBED_DIM})
        )
    """)

    batch = []
    for i in range(n):
        v = rng.randn(EMBED_DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        batch.append((v,))
        if len(batch) >= 200:
            conn.executemany("INSERT INTO vectors (embedding) VALUES (%s)", batch)
            batch = []
    if batch:
        conn.executemany("INSERT INTO vectors (embedding) VALUES (%s)", batch)

    return query


def _timed_search(conn, sql: str, params: tuple) -> tuple[float, list]:
    start = time.perf_counter()
    rows = conn.execute(sql, params).fetchall()
    elapsed = (time.perf_counter() - start) * 1000
    return elapsed, rows


def run(conn) -> None:
    print("\n--- HNSW vs Exact Search ---")
    print("Demonstrates: index on/off, recall overlap, latency\n")

    reset_schema(conn)
    n_rows = 2000
    print(f"Seeding {n_rows} random unit vectors...")
    query = _seed_data(conn, n_rows)

    sql = "SELECT id FROM vectors ORDER BY embedding <=> %s LIMIT 10"
    params = (query,)

    # Exact: disable index scan
    conn.execute("SET enable_indexscan = off")
    exact_ms, exact_ids = _timed_search(conn, sql, params)
    exact_ids = [r[0] for r in exact_ids]

    # Approximate: no index yet (still seq scan)
    conn.execute("SET enable_indexscan = on")
    no_index_ms, _ = _timed_search(conn, sql, params)

    # Build HNSW
    print("Building HNSW index...")
    t0 = time.perf_counter()
    conn.execute("""
        CREATE INDEX vectors_hnsw ON vectors
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)
    build_ms = (time.perf_counter() - t0) * 1000
    conn.execute("ANALYZE vectors")

    for ef in [40, 100, 200]:
        conn.execute(f"SET hnsw.ef_search = {ef}")
        ann_ms, ann_ids = _timed_search(conn, sql, params)
        ann_ids = [r[0] for r in ann_ids]
        overlap = len(set(exact_ids) & set(ann_ids))
        print(f"  ef_search={ef}: {ann_ms:.1f}ms, recall@{len(exact_ids)}={overlap}/{len(exact_ids)}")

    conn.execute("RESET enable_indexscan")
    conn.execute("RESET hnsw.ef_search")

    rows = [
        ("Exact (index off)", f"{exact_ms:.1f}"),
        ("No HNSW index", f"{no_index_ms:.1f}"),
        ("HNSW build time", f"{build_ms:.0f}"),
    ]
    print("\nTiming summary (ms):")
    print_results(rows, ["mode", "milliseconds"])

    plan = conn.execute(
        "EXPLAIN SELECT id FROM vectors ORDER BY embedding <=> %s LIMIT 10",
        (query,),
    ).fetchall()
    print("\nPlan with HNSW (look for 'Index Scan using vectors_hnsw'):")
    for row in plan[:4]:
        print(f"  {row[0]}")
