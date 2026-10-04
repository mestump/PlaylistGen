"""Spotify Extended Streaming History (Streaming_History_Audio_*.json) -> play_events."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from ..db import get_meta, set_meta
from ..resolve import upsert_track


def _parse_ts(v) -> int | None:
    if not v:
        return None
    s = str(v).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        try:
            dt = datetime.strptime(str(v).strip(), "%Y-%m-%d %H:%M")
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _rows(path: Path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list")
    for r in data:
        if isinstance(r, dict):
            yield r


def _extract(r: dict) -> tuple[str, str, int | None, int, int] | None:
    artist = r.get("master_metadata_album_artist_name") or r.get("artistName")
    title = r.get("master_metadata_track_name") or r.get("trackName")
    if not artist or not title:
        return None
    ts = _parse_ts(r.get("ts") or r.get("endTime"))
    ms = int(r.get("msPlayed") or 0)
    skipped = 1 if r.get("skipped") else 0
    return str(artist), str(title), ts, ms, skipped


def ingest_spotify_history(con: sqlite3.Connection, paths: list[str | Path]) -> dict:
    files = sorted(Path(p) for p in paths)
    total_events = 0
    report = []
    for path in files:
        if not path.exists():
            raise FileNotFoundError(path)
        stamp_key = f"ingest:spotify-history:{path}:{int(path.stat().st_mtime)}"
        if get_meta(con, stamp_key):
            report.append({"path": path.name, "events": 0, "skipped": "unchanged"})
            continue
        batch = []
        for r in _rows(path):
            x = _extract(r)
            if x is None:
                continue
            artist, title, ts, ms, skipped = x
            tid = upsert_track(con, artist, title)
            batch.append((tid, "spotify", ts, ms, skipped))
        con.executemany(
            "INSERT INTO play_events(track_id,source,played_at,ms_played,skipped) VALUES(?,?,?,?,?)",
            batch,
        )
        set_meta(con, stamp_key, "1")
        total_events += len(batch)
        report.append({"path": path.name, "events": len(batch)})
    con.commit()
    return {"source": "spotify-history", "events": total_events, "files": report}
