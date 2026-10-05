"""pgen CLI: ingest / status. Build & make land in M2/M3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import db as dbmod
from .ingest.files import DEFAULT_LIBRARY, ingest_directory
from .ingest.ipod import ingest_ipod
from .ingest.strawberry import DEFAULT_DB as STRAWBERRY_DB
from .ingest.strawberry import ingest_strawberry
from .ingest.spotify_history import ingest_spotify_history

DEFAULT_DB_PATH = Path.home() / ".local/share/pgen/v2.db"
IPOD_ITDB = Path.home() / "run"  # placeholder; resolved via --ipod-itdb or autodetect


def _autodetect_ipod_itdb() -> Path | None:
    base = Path("/run/media")
    if not base.exists():
        return None
    for user_dir in sorted(base.glob("*/")):
        for name in ("MICHAEL_S I", "IPOD", "IPOD*"):
            for hit in user_dir.glob(name):
                cand = hit / "iPod_Control/iTunes/iTunesDB"
                if cand.exists():
                    return cand
    return None


def cmd_ingest(args) -> int:
    con = dbmod.connect(args.db)
    try:
        if getattr(args, "force", False):
            con.execute("DELETE FROM meta WHERE key LIKE 'ingest:%:stamp'")
            con.commit()
        results = []
        want = args.sources
        if "files" in want:
            results.append(ingest_directory(con, args.library))
        if "strawberry" in want:
            results.append(ingest_strawberry(con, args.strawberry_db))
        if "ipod" in want:
            sb = Path(args.ipod_sb_db) if getattr(args, "ipod_sb_db", None) else Path(args.strawberry_db)
            if not sb.exists():
                print(f"strawberry db not found: {sb}", file=sys.stderr)
                return 2
            results.append(ingest_ipod(con, sb))
        if "spotify-history" in want:
            paths: list[str | Path] = [Path(p) for p in args.spotify_glob]
            if not paths:
                print("no history files match; pass --spotify-glob", file=sys.stderr)
                return 2
            results.append(ingest_spotify_history(con, paths))
        print(json.dumps(results, indent=2, default=str))
        return 0
    finally:
        counts = con.execute(
            "SELECT (SELECT COUNT(*) FROM tracks), (SELECT COUNT(*) FROM play_events), "
            "(SELECT COALESCE(SUM(plays),0) FROM source_counts)"
        ).fetchone()
        print(f"totals: tracks={counts[0]} events={counts[1]} counter_plays={counts[2]}")
        con.close()


def cmd_make(args) -> int:
    from . import generate

    con = dbmod.connect(args.db)
    res = generate.generate(con, args.n, args.seed)
    print(json.dumps(res, indent=2))
    return 0 if res["picked"] else 1


def cmd_status(args) -> int:
    con = dbmod.connect(args.db)
    try:
        t = con.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
        ev = con.execute(
            "SELECT source, COUNT(*), MIN(played_at), MAX(played_at) FROM play_events GROUP BY source"
        ).fetchall()
        cs = con.execute(
            "SELECT source, COUNT(*), SUM(plays), SUM(skips) FROM source_counts GROUP BY source"
        ).fetchall()
        print(f"tracks: {t}")
        for src, n, mn, mx in ev:
            print(f"events[{src}]: {n:,} span {mn}..{mx}")
        for src, n, pl, sk in cs:
            print(f"counts[{src}]: tracks={n:,} plays={pl:,} skips={sk:,}")
        return 0
    finally:
        con.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pgen")
    ap.add_argument("--db", default=str(DEFAULT_DB_PATH))
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest")
    p.add_argument("sources", nargs="+", choices=["files", "strawberry", "ipod", "spotify-history", "all"])
    p.add_argument("--library", default=DEFAULT_LIBRARY)
    p.add_argument("--strawberry-db", default=str(STRAWBERRY_DB))
    p.add_argument("--ipod-sb-db", default=None, help="override strawberry db for ipod counts")
    p.add_argument("--spotify-glob", nargs="*", default=[])
    p.add_argument("--force", action="store_true", help="re-ingest even if the source stamp is unchanged")
    p.set_defaults(func=cmd_ingest)

    s = sub.add_parser("status")
    s.set_defaults(func=cmd_status)

    g = sub.add_parser("make")
    g.add_argument("-n", type=int, default=30)
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(func=cmd_make)

    args = ap.parse_args(argv)
    if "all" in getattr(args, "sources", []):
        args.sources = ["files", "strawberry", "ipod", "spotify-history"]
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
