"""iPod source: parse iTunesDB (mhit records) directly. Device files are authoritative;
Strawberry's device cache has been observed stale (lastseen=0).

Field offsets calibrated empirically 2026-10-04 against this device's DB (14,128 mhits)
with strawberry device_1_songs as reference:
  +0x30 playcount  (72% monotone-match vs cache on pc>=10 rows; +0x2C only 7%)
  +0x2C skipcount  (provisional: 98.7% nonzero, median 7, sum 0.6x plays - plausible; unvalidated against cache)
  +0x18 time added (100% tslike) - NOT lastplayed; iPod lastplayed deferred (Play Counts mhdp, M2)
mhod strings: utf-16-le at +0x18..+totlen; type 1=title 2=path 4=artist; path sep ':'.
"""

from __future__ import annotations

import struct
import sqlite3
from pathlib import Path

from ..db import get_meta, set_meta
from ..resolve import upsert_counts, upsert_track

U32 = struct.Struct("<I")

MHSD_TRACK = 1
MHOD_TITLE, MHOD_PATH, MHOD_ARTIST = 1, 2, 4
PLAYS_OFF, SKIPS_OFF = 0x30, 0x2C


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
                        "plays": _u32(data, p + PLAYS_OFF),
                        "skips": _u32(data, p + SKIPS_OFF),
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


def ingest_ipod(con: sqlite3.Connection, itdb_path: str | Path) -> dict:
    itdb_path = Path(itdb_path)
    if not itdb_path.exists():
        raise FileNotFoundError(f"iTunesDB not found: {itdb_path}")
    stamp = f"ipod:{itdb_path}:{int(itdb_path.stat().st_mtime)}"
    key = "ingest:ipod:stamp"
    if get_meta(con, key) == stamp:
        return {"source": "ipod", "skipped": "unchanged"}

    tracks = parse_itdb(itdb_path)
    acc: dict[int, list[int]] = {}
    n = 0
    for t in tracks:
        tid = upsert_track(con, t["artist"], t["title"])
        a = acc.setdefault(tid, [0, 0])
        a[0] += t["plays"]
        a[1] += t["skips"]
        n += 1
    plays = skips = 0
    skiplte = 0
    for tid, (p, s) in acc.items():
        if p or s:
            upsert_counts(con, tid, "ipod", p, s, None)
        plays += p
        skips += s
        if s <= p:
            skiplte += 1
    set_meta(con, key, stamp)
    con.commit()
    return {
        "source": "ipod",
        "records": n,
        "tracks": len(acc),
        "plays": plays,
        "skips": skips,
        "skips_le_plays_pct": round(100 * skiplte / max(len(acc), 1), 1),
        "stamp": stamp,
    }
