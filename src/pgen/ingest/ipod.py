"""iPod source.

Counts: Strawberry's device_1_songs table (libgpod-grade, written at last sync) is the
per-track source of truth for play/skip counts -- ~5,965 plays on this device. The raw
iTunesDB parser below is kept for identity/path use and calibration notes; its heuristic
count offsets (mhit+0x30 etc.) over-counted ~36x and are NOT used for totals.
Open question (TODO M3+): 'Play Counts' file = 14,128 x 28B entries (mhdp hdr: entry
size 28, count == mhit count); entry-order convention unresolved; would give the
post-sync delta. iTunesDB mhit count offsets +0x5C/+0x60 also unresolved (eq ~52%).

mhod strings: utf-16-le; type 1=title 2=path 4=artist; path sep ':'.
"""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

from ..db import get_meta, set_meta
from ..resolve import upsert_counts, upsert_track

U32 = struct.Struct("<I")

MHSD_TRACK = 1
MHOD_TITLE, MHOD_PATH, MHOD_ARTIST = 1, 2, 4


def _u32(buf: bytes, off: int) -> int:
    return U32.unpack_from(buf, off)[0]


def _mhod_str(buf: bytes, rec: int, tot: int, want: int) -> str | None:
    q = rec + _u32(buf, rec + 4)
    end = rec + tot
    while q + 0x28 <= end:
        if buf[q:q + 4] == b"mhod":
            tl = _u32(buf, q + 8)
            if _u32(buf, q + 0x0C) == want:
                # string body: u32 unk, u32 len, u32 unk, u32 unk, then utf-16-le text
                s = buf[q + 0x28:q + tl].decode("utf-16-le", errors="replace").strip("\x00")
                return s or None
            q += max(tl, 0x18)
        else:
            q += 4
    return None


def parse_itdb(path: str | Path) -> list[dict]:
    """Parse mhit identity (title/artist/path). Counts NOT included -- see module docstring."""
    data = Path(path).read_bytes()
    j = data.find(b"mhsd")
    while j >= 0 and _u32(data, j + 0x0C) != MHSD_TRACK:
        j = data.find(b"mhsd", j + 4)
    if j < 0:
        raise ValueError("no track mhsd in iTunesDB")
    start = j + _u32(data, j + 4)
    if data[start:start + 4] != b"mhlt":
        raise ValueError("expected mhlt after track mhsd")
    end = j + _u32(data, j + 8)

    out: list[dict] = []
    p = start + _u32(data, start + 4)
    bad = 0
    while p < end - 0x18:
        if data[p:p + 4] == b"mhit":
            tot = _u32(data, p + 8)
            title = _mhod_str(data, p, tot, MHOD_TITLE)
            artist = _mhod_str(data, p, tot, MHOD_ARTIST)
            if title and artist:
                out.append(
                    {
                        "title": title,
                        "artist": artist,
                        "device_path": _mhod_str(data, p, tot, MHOD_PATH),
                    }
                )
            else:
                bad += 1
            p += max(tot, 0x18)
        else:
            p += 4
    if out and bad > len(out):
        raise ValueError(f"mhod parse failure rate too high: {bad} bad / {len(out)} good")
    return out


def ingest_ipod(con: sqlite3.Connection, sb_db: str | Path) -> dict:
    """iPod play/skip counts from strawberry's device cache (same device, libgpod-parsed)."""
    sb_db = Path(sb_db)
    if not sb_db.exists():
        raise FileNotFoundError(f"strawberry db not found: {sb_db}")
    stamp = f"ipod-cache:{int(sb_db.stat().st_mtime)}"
    key = "ingest:ipod:stamp"
    if get_meta(con, key) == stamp:
        return {"source": "ipod", "skipped": "unchanged"}

    con.execute(f"ATTACH DATABASE '{sb_db}' AS sbdev")
    rows = con.execute(
        "SELECT artist, title, playcount, skipcount, lastplayed FROM sbdev.device_1_songs "
        "WHERE artist IS NOT NULL AND TRIM(artist)<>'' AND title IS NOT NULL AND TRIM(title)<>''"
    ).fetchall()
    con.execute("DETACH DATABASE sbdev")

    acc: dict[int, list[int]] = {}
    for artist, title, pc, sk, _lp in rows:
        tid = upsert_track(con, artist, title)
        a = acc.setdefault(tid, [0, 0])
        a[0] += int(pc or 0)
        a[1] += int(sk or 0)
    plays = skips = 0
    for tid, (p, s) in acc.items():
        if p or s:
            upsert_counts(con, tid, "ipod", p, s, None)
        plays += p
        skips += s
    set_meta(con, key, stamp)
    con.commit()
    return {
        "source": "ipod",
        "records": len(rows),
        "tracks": len(acc),
        "plays": plays,
        "skips": skips,
        "stamp": stamp,
    }
