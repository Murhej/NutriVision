"""Initialize NutriVision database schema from SQL file.

Usage:
  python scripts/init_nutrivision_db.py
  python scripts/init_nutrivision_db.py --db outputs/nutrivision.db
  python scripts/init_nutrivision_db.py --schema src/api/database_schema.sql
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize NutriVision SQLite schema")
    parser.add_argument("--db", default="outputs/nutrivision.db", help="SQLite database file path")
    parser.add_argument("--schema", default="src/api/database_schema.sql", help="Path to SQL schema file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    db_path = Path(args.db)
    schema_path = Path(args.schema)

    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    sql = schema_path.read_text(encoding="utf-8")

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(sql)
        conn.commit()

    print(f"Schema initialized successfully: {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
