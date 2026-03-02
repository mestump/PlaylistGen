#!/usr/bin/env python3
"""
Load enrichment JSON files into the PlaylistGen SQLite enrichment cache.

Each JSON file should be a list of records with at minimum a cache key and
mood/energy/valence values.  Two record shapes are accepted:

  Cache-dump format (exported from another run):
    {"key": "artist - track", "mood": "Happy", "energy": 7, "valence": 8,
     "tags": ["upbeat", "pop"]}

  Artist/track format (hand-crafted or from a spreadsheet export):
    {"artist": "Daft Punk", "track": "Get Lucky", "mood": "Groovy",
     "energy": 8, "valence": 9, "tags": ["funk", "disco"]}

Usage:
    python scripts/load_enrichment_cache.py <folder>
    python scripts/load_enrichment_cache.py <folder> --db ~/.playlistgen/claude_enrichment.sqlite
    python scripts/load_enrichment_cache.py <folder> --dry-run
"""

import argparse
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

DEFAULT_DB = Path.home() / ".playlistgen" / "claude_enrichment.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS claude_enrichment (
    key         TEXT PRIMARY KEY,
    mood        TEXT,
    energy      INTEGER,
    valence     INTEGER,
    tags        TEXT,
    enriched_at INTEGER
);
"""


def init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(_SCHEMA)
    conn.commit()
    return conn


def parse_record(raw: dict) -> tuple[str, dict] | None:
    """
    Return (cache_key, fields) from a raw JSON record, or None if unusable.
    """
    # Determine the cache key
    if "key" in raw:
        key = str(raw["key"]).strip().lower()
    elif "artist" in raw and "track" in raw:
        artist = str(raw.get("artist") or "").strip()
        track = str(raw.get("track") or "").strip()
        if not artist or not track:
            return None
        key = f"{artist} - {track}".lower()
    else:
        return None

    if not key:
        return None

    mood = str(raw.get("mood") or "").strip() or None
    energy = raw.get("energy")
    valence = raw.get("valence")
    tags = raw.get("tags")

    try:
        energy = int(energy) if energy is not None else None
    except (ValueError, TypeError):
        energy = None

    try:
        valence = int(valence) if valence is not None else None
    except (ValueError, TypeError):
        valence = None

    tags_json = json.dumps(tags) if isinstance(tags, list) else None

    return key, {
        "mood": mood,
        "energy": energy,
        "valence": valence,
        "tags": tags_json,
    }


def load_file(path: Path, conn: sqlite3.Connection, dry_run: bool) -> tuple[int, int]:
    """Load one JSON file. Returns (loaded, skipped) counts."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logging.warning("  Skipping %s — could not parse JSON: %s", path.name, exc)
        return 0, 0

    if isinstance(data, dict):
        # Single record
        data = [data]

    if not isinstance(data, list):
        logging.warning("  Skipping %s — expected a JSON array or object.", path.name)
        return 0, 0

    loaded = skipped = 0
    now = int(time.time())

    for raw in data:
        if not isinstance(raw, dict):
            skipped += 1
            continue

        result = parse_record(raw)
        if result is None:
            skipped += 1
            continue

        key, fields = result

        if not dry_run:
            conn.execute(
                """INSERT OR REPLACE INTO claude_enrichment
                   (key, mood, energy, valence, tags, enriched_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    key,
                    fields["mood"],
                    fields["energy"],
                    fields["valence"],
                    fields["tags"],
                    now,
                ),
            )
        loaded += 1

    if not dry_run:
        conn.commit()

    return loaded, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load enrichment JSON files into the PlaylistGen cache."
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Directory containing enrichment JSON files.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help=f"SQLite cache path (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate files without writing to the database.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show per-file progress.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    folder: Path = args.folder.expanduser().resolve()
    if not folder.is_dir():
        logging.error("'%s' is not a directory.", folder)
        sys.exit(1)

    json_files = sorted(folder.glob("*.json"))
    if not json_files:
        logging.error("No .json files found in '%s'.", folder)
        sys.exit(1)

    if args.dry_run:
        logging.info("DRY RUN — no data will be written.")
        conn = None
    else:
        db_path: Path = args.db.expanduser().resolve()
        logging.info("Cache DB: %s", db_path)
        conn = init_db(db_path)

    total_loaded = total_skipped = 0

    for json_file in json_files:
        logging.info("Loading %s …", json_file.name)
        loaded, skipped = load_file(json_file, conn, dry_run=args.dry_run)
        total_loaded += loaded
        total_skipped += skipped
        logging.info("  %d records loaded, %d skipped.", loaded, skipped)

    if conn:
        conn.close()

    prefix = "[DRY RUN] " if args.dry_run else ""
    logging.info(
        "%sDone — %d file(s), %d records loaded, %d skipped.",
        prefix,
        len(json_files),
        total_loaded,
        total_skipped,
    )


if __name__ == "__main__":
    main()
