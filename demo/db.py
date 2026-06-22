"""Database connection and schema helpers."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Generator

import psycopg
from pgvector.psycopg import register_vector

DEFAULT_DSN = "postgresql://pgvector:pgvector@localhost:5433/pgvector_demo"
EMBED_DIM = 384  # all-MiniLM-L6-v2


def get_dsn() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DSN)


@contextmanager
def connect() -> Generator[psycopg.Connection, None, None]:
    conn = psycopg.connect(get_dsn())
    register_vector(conn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def wait_for_db(timeout: float = 60.0, interval: float = 2.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with connect() as conn:
                conn.execute("SELECT 1")
            return True
        except psycopg.OperationalError:
            time.sleep(interval)
    return False


def reset_schema(conn: psycopg.Connection) -> None:
    conn.execute("DROP SCHEMA IF EXISTS demo CASCADE")
    conn.execute("CREATE SCHEMA demo")
    conn.execute("SET search_path TO demo, public")
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")


def init_extensions(conn: psycopg.Connection) -> None:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute("SET search_path TO demo, public")


def explain_query(conn: psycopg.Connection, sql: str, *params) -> str:
    cur = conn.execute(f"EXPLAIN (ANALYZE, BUFFERS) {sql}", params)
    return "\n".join(row[0] for row in cur.fetchall())


def print_results(rows: list, columns: list[str] | None = None) -> None:
    if not rows:
        print("  (no results)")
        return
    if columns is None:
        columns = [f"col{i}" for i in range(len(rows[0]))]
    widths = [max(len(c), max(len(str(r[i])) for r in rows)) for i, c in enumerate(columns)]
    header = " | ".join(c.ljust(widths[i]) for i, c in enumerate(columns))
    print(f"  {header}")
    print(f"  {'-' * len(header)}")
    for row in rows:
        print("  " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(row))))
