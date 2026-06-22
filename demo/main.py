#!/usr/bin/env python3
"""
pgvector interactive demo — menu-driven use case runner with Docker Postgres.

Usage:
    python main.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Ensure demo package imports resolve when run from demo/
DEMO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO_DIR))

from db import connect, get_dsn, init_extensions, reset_schema, wait_for_db
from embeddings import embedding_backend
from use_cases import USE_CASES

COMPOSE_FILE = DEMO_DIR / "docker-compose.yml"


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def pause() -> None:
    input("\nPress Enter to continue...")


def run_command(cmd: list[str], cwd: Path | None = None) -> int:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=cwd or DEMO_DIR)


def docker_available() -> bool:
    return shutil.which("docker") is not None


def compose_cmd(*args: str) -> list[str]:
    return ["docker", "compose", "-f", str(COMPOSE_FILE), *args]


def docker_start() -> None:
    if not docker_available():
        print("  Docker not found. Install Docker Desktop or start Postgres manually.")
        print(f"  Set DATABASE_URL if not using default: {get_dsn()}")
        pause()
        return
    print("\nStarting pgvector container...")
    code = run_command(compose_cmd("up", "-d"))
    if code != 0:
        print("  Failed to start container.")
        pause()
        return
    print("  Waiting for database...")
    if wait_for_db():
        print("  Database is ready.")
    else:
        print("  Database did not become ready in time.")
    pause()


def docker_stop() -> None:
    if not docker_available():
        print("  Docker not available.")
        pause()
        return
    print("\nStopping pgvector container...")
    run_command(compose_cmd("down"))
    print("  Container stopped.")
    pause()


def docker_logs() -> None:
    if not docker_available():
        print("  Docker not available.")
        pause()
        return
    run_command(compose_cmd("logs", "--tail", "50"))
    pause()


def docker_reset() -> None:
    if not docker_available():
        print("  Docker not available.")
        pause()
        return
    confirm = input("  This removes the container AND volume (all data). Continue? [y/N] ").strip().lower()
    if confirm != "y":
        return
    run_command(compose_cmd("down", "-v"))
    print("  Volume removed.")
    pause()


def init_database() -> None:
    print("\nInitializing database extensions...")
    try:
        with connect() as conn:
            conn.execute("CREATE SCHEMA IF NOT EXISTS demo")
            init_extensions(conn)
        print("  Extension 'vector' enabled. Schema 'demo' ready.")
    except Exception as e:
        print(f"  Error: {e}")
        print("  Is the container running? Use menu option [S] first.")
    pause()


def reset_database() -> None:
    confirm = input("  Drop and recreate demo schema? [y/N] ").strip().lower()
    if confirm != "y":
        return
    try:
        with connect() as conn:
            reset_schema(conn)
        print("  Demo schema reset.")
    except Exception as e:
        print(f"  Error: {e}")
    pause()


def run_use_case(key: str) -> None:
    label, fn = USE_CASES[key]
    try:
        with connect() as conn:
            conn.execute("SET search_path TO demo, public")
            fn(conn)
    except Exception as e:
        print(f"\n  Error running demo: {e}")
        import traceback
        traceback.print_exc()
    pause()


def run_all_use_cases() -> None:
    confirm = input("  Run all 9 use cases sequentially? [y/N] ").strip().lower()
    if confirm != "y":
        return
    for key in sorted(USE_CASES.keys(), key=int):
        label, fn = USE_CASES[key]
        print(f"\n{'=' * 60}")
        print(f"  Running: {label}")
        print("=" * 60)
        try:
            with connect() as conn:
                conn.execute("SET search_path TO demo, public")
                fn(conn)
        except Exception as e:
            print(f"  Error: {e}")
        input("  Press Enter for next demo...")
    print("\n  All demos complete.")


def print_menu() -> None:
    clear_screen()
    print("=" * 60)
    print("  pgvector Interactive Demo")
    print("=" * 60)
    print(f"  Database : {get_dsn()}")
    print(f"  Embeddings: {embedding_backend()}")
    print("-" * 60)
    print("  Docker")
    print("    [S] Start container")
    print("    [T] Stop container")
    print("    [L] View logs")
    print("    [R] Reset container + volume")
    print("-" * 60)
    print("  Database")
    print("    [I] Initialize extensions")
    print("    [D] Reset demo schema")
    print("-" * 60)
    print("  Use Cases")
    for key, (label, _) in sorted(USE_CASES.items(), key=lambda x: int(x[0])):
        print(f"    [{key}] {label}")
    print("    [A] Run ALL use cases")
    print("-" * 60)
    print("    [Q] Quit")
    print("=" * 60)


def main() -> None:
    # Load .env if present
    env_file = DEMO_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    handlers = {
        "s": docker_start,
        "t": docker_stop,
        "l": docker_logs,
        "r": docker_reset,
        "i": init_database,
        "d": reset_database,
        "a": run_all_use_cases,
        "q": lambda: sys.exit(0),
    }
    handlers.update({k: lambda k=k: run_use_case(k) for k in USE_CASES})

    print("pgvector Demo — ensure Docker is running, then press [S] to start.")

    while True:
        print_menu()
        choice = input("\n  Select option: ").strip().lower()
        if not choice:
            continue
        handler = handlers.get(choice)
        if handler:
            handler()
        else:
            print("  Invalid option.")
            pause()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Bye.")
