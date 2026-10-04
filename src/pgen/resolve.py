"""Identity resolution: every source funnels through upsert_track. First tag wins for display fields."""

from __future__ import annotations

import sqlite3

from .normalize import track_key


def upsert_track(
    con: sqlite3.Connection,
    artist: str | None,
    title: str | None,
    *,
    album: str | None = None,
    year: int | None = None,
    path: str | None = None,
    path_source: str | None = None,
    mtime: float | None = None,
    size: int | None = None,
) -> int:
    a, t = track_key(artist, title)
    if not a and not t:
        raise ValueError("empty artist and title")
    cur = con.execute(
        "SELECT id, artist, title, album, year FROM tracks WHERE artist_canon=? AND title_canon=?",
        (a, t),
    )
    row = cur.fetchone()
    if row is None:
        cur = con.execute(
            "INSERT INTO tracks(artist_canon,title_canon,artist,title,album,year) VALUES(?,?,?,?,?,?)",
            (a, t, artist, title, album, year),
        )
        track_id = int(cur.lastrowid)  # type: ignore[arg-type]
    else:
        track_id = row["id"]
        if artist and not row["artist"]:
            con.execute("UPDATE tracks SET artist=? WHERE id=?", (artist, track_id))
        if title and not row["title"]:
            con.execute("UPDATE tracks SET title=? WHERE id=?", (title, track_id))
        if album and not row["album"]:
            con.execute("UPDATE tracks SET album=? WHERE id=?", (album, track_id))
        if year and not row["year"]:
            con.execute("UPDATE tracks SET year=? WHERE id=?", (year, track_id))
    if path and path_source:
        con.execute(
            "INSERT INTO track_paths(track_id,path,source,mtime,size) VALUES(?,?,?,?,?) "
            "ON CONFLICT(path) DO UPDATE SET track_id=excluded.track_id, mtime=excluded.mtime",
            (track_id, path, path_source, mtime, size),
        )
    return track_id


def upsert_counts(
    con: sqlite3.Connection, track_id: int, source: str, plays: int, skips: int, lastplayed: int | None
) -> None:
    con.execute(
        "INSERT INTO source_counts(track_id,source,plays,skips,lastplayed) VALUES(?,?,?,?,?) "
        "ON CONFLICT(track_id,source) DO UPDATE SET plays=excluded.plays, skips=excluded.skips, "
        "lastplayed=excluded.lastplayed",
        (track_id, source, max(0, int(plays)), max(0, int(skips)), lastplayed),
    )
