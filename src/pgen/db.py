"""SQLite store. Append-only events + counter tables; everything else is a rebuildable projection."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks(
  id INTEGER PRIMARY KEY,
  artist_canon TEXT NOT NULL,
  title_canon TEXT NOT NULL,
  artist TEXT, title TEXT, album TEXT, year INTEGER,
  first_seen TEXT DEFAULT (datetime('now')),
  UNIQUE(artist_canon, title_canon)
);
CREATE TABLE IF NOT EXISTS track_paths(
  track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
  path TEXT NOT NULL UNIQUE,
  source TEXT NOT NULL,
  mtime REAL, size INTEGER
);
CREATE INDEX IF NOT EXISTS idx_paths_track ON track_paths(track_id);
CREATE TABLE IF NOT EXISTS play_events(
  id INTEGER PRIMARY KEY,
  track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
  source TEXT NOT NULL,
  played_at INTEGER,            -- unix UTC
  ms_played INTEGER,
  skipped INTEGER,              -- 0/1/NULL unknown
  session_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_track ON play_events(track_id);
CREATE INDEX IF NOT EXISTS idx_events_time ON play_events(played_at);
CREATE TABLE IF NOT EXISTS source_counts(
  track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
  source TEXT NOT NULL,         -- strawberry | ipod | itunes_legacy
  plays INTEGER NOT NULL DEFAULT 0,
  skips INTEGER NOT NULL DEFAULT 0,
  lastplayed INTEGER,
  PRIMARY KEY(track_id, source)
);
CREATE TABLE IF NOT EXISTS features(
  track_id INTEGER PRIMARY KEY REFERENCES tracks(id) ON DELETE CASCADE,
  model TEXT, dim INTEGER, vec BLOB,
  scalars TEXT,                 -- json: tempo/energy/danceability/valence/popularity...
  source TEXT,                  -- spotify-parquet | local
  updated TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS runs(
  id INTEGER PRIMARY KEY,
  kind TEXT, seed INTEGER, params TEXT,
  created TEXT DEFAULT (datetime('now')),
  m3u_path TEXT
);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p), timeout=60)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=60000")
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript(SCHEMA)
    return con


def get_meta(con: sqlite3.Connection, key: str) -> str | None:
    row = con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(con: sqlite3.Connection, key: str, value: str) -> None:
    con.execute(
        "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
