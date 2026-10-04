"""Strawberry library source (read-only). Counters go to source_counts, never fabricated events."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ..db import get_meta, set_meta
from ..resolve import upsert_counts, upsert_track

DEFAULT_DB = Path.home() / ".local/share/strawberry/strawberry/strawberry.db"


def _attach(con: sqlite3.Connection, path: str, alias: str = "sb") -> None:
    con.execute(f"ATTACH DATABASE '{path}' AS {alias}")


def ingest_strawberry(con: sqlite3.Connection, sb_path: str | Path | None = None) -> dict:
    sb_path = Path(sb_path or DEFAULT_DB)
    if not sb_path.exists():
        raise FileNotFoundError(f"strawberry db not found: {sb_path}")
    stamp = f"strawberry:{int(sb_path.stat().st_mtime)}"
    prev = get_meta(con, "ingest:strawberry:stamp")
    if prev == stamp:
        return {"source": "strawberry", "skipped": "unchanged", "stamp": stamp}

    main = con
    _attach(main, str(sb_path))
    rows = main.execute(
        "SELECT artist, title, album, year, playcount, skipcount, lastplayed "
        "FROM sb.songs WHERE artist IS NOT NULL AND title IS NOT NULL"
    ).fetchall()
    n = plays = skips = 0
    for r in rows:
        tid = upsert_track(
            main, r["artist"], r["title"], album=r["album"], year=r["year"]
        )
        pc, sc = int(r["playcount"] or 0), int(r["skipcount"] or 0)
        if pc or sc:
            upsert_counts(main, tid, "strawberry", pc, sc, r["lastplayed"])
            plays += pc
            skips += sc
        n += 1
    main.execute("DETACH DATABASE sb")
    set_meta(main, "ingest:strawberry:stamp", stamp)
    main.commit()
    return {"source": "strawberry", "tracks": n, "plays": plays, "skips": skips, "stamp": stamp}
