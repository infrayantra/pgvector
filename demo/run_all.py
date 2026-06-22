#!/usr/bin/env python3
"""Run all use case demos non-interactively (for CI or quick smoke test)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from db import connect, wait_for_db
from use_cases import USE_CASES


def main() -> int:
    if not wait_for_db(timeout=30):
        print("ERROR: Database not available. Run: docker compose up -d")
        return 1

    failed = 0
    for key in sorted(USE_CASES.keys(), key=int):
        label, fn = USE_CASES[key]
        print(f"\n{'=' * 60}\n[{key}] {label}\n{'=' * 60}")
        try:
            with connect() as conn:
                conn.execute("SET search_path TO demo, public")
                fn(conn)
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

    print(f"\nDone: {len(USE_CASES) - failed}/{len(USE_CASES)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
