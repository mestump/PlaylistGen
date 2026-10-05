"""Ingest Spotify artist genres per library track (via the match-cache bridge).

track_genre(track_id, genre) -- raw spotify genre tags, one row per (track,tag).
The tag vocabulary is noisy; consumers jaccard it, no taxonomy needed.
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

PD = Path("/mnt/storage/spotify-data")
CACHE = Path.home() / "pgen-tmp/match-cache.parquet"


def run(db_path: str) -> int:
    import duckdb

    con = sqlite3.connect(db_path)
    con.execute("DROP TABLE IF EXISTS track_genre")
    con.execute(
        "CREATE TABLE track_genre(track_id INTEGER NOT NULL, genre TEXT NOT NULL, "
        "PRIMARY KEY(track_id,genre))"
    )
    con.execute("CREATE INDEX IF NOT EXISTS tg_track ON track_genre(track_id)")

    q = f"""
        SELECT m.tid AS tid, g.genre AS genre
        FROM read_parquet('{CACHE}') m
        JOIN read_parquet('{PD}/tracks.parquet', union_by_name=true) t ON t.id = m.spotify_id
        JOIN read_parquet('{PD}/track_artists.parquet', union_by_name=true) ta ON ta.track_rowid = t.rowid
        JOIN read_parquet('{PD}/artist_genres.parquet', union_by_name=true) g ON g.artist_rowid = ta.artist_rowid
        WHERE g.genre IS NOT NULL AND trim(g.genre) <> ''
        GROUP BY m.tid, g.genre
    """
    rows = duckdb.sql(q).fetchall()
    con.executemany("INSERT OR IGNORE INTO track_genre VALUES(?,?)", rows)
    con.commit()
    n_tracks = con.execute("SELECT COUNT(DISTINCT track_id) FROM track_genre").fetchone()[0]
    total = con.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
    print({"genre_rows": len(rows), "tracks_with_genres": n_tracks, "tracks_total": total})
    con.close()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pgen-ingest-genres")
    ap.add_argument("--db", default=str(Path.home() / ".local/share/pgen/v2.db"))
    args = ap.parse_args(argv)
    return run(args.db)


if __name__ == "__main__":
    raise SystemExit(main())
