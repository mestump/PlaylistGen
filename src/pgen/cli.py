"""pgen CLI: ingest / status. Build & make land in M2/M3."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import db as dbmod
from .ingest.files import ingest_directory
from .ingest.ipod import ingest_ipod
from .ingest.strawberry import DEFAULT_DB as STRAWBERRY_DB
from .ingest.strawberry import ingest_strawberry
from .ingest.spotify_history import ingest_spotify_history

DEFAULT_DB_PATH = Path.home() / ".local/share/pgen/v2.db"
DEFAULT_LIBRARY = Path.home() / "Music"
PARQUET_DIR_DEFAULT = os.environ.get("PGEN_PARQUET_DIR", "")


def _autodetect_ipod_itdb() -> Path | None:
    base = Path("/run/media")
    if not base.exists():
        return None
    for itdb in sorted(base.glob("*/*/iPod_Control/iTunes/iTunesDB")):
        return itdb
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
    res = generate.generate(con, args.n, args.seed, getattr(args, "scene", None))
    print(json.dumps(res, indent=2))
    return 0 if res["picked"] else 1


def cmd_build(args) -> int:
    from . import build_affinity, build_signals

    con = dbmod.connect(args.db)
    try:
        stats = build_signals.rebuild(con)
        print("signals:", stats)
        if not args.skip_affinity:
            n_sess = build_affinity.sessionize(con)
            aff = build_affinity.affinity(con)
            print({"sessions": n_sess, **aff})
        con.commit()
        return 0
    finally:
        con.close()


def cmd_genres(args) -> int:
    from . import ingest_genres

    if not args.parquet_dir:
        print("no parquet dir: pass --parquet-dir or set PGEN_PARQUET_DIR", file=sys.stderr)
        return 2
    return ingest_genres.run(args.db, args.parquet_dir)


def cmd_features(args) -> int:
    from . import enrich_features

    return enrich_features.run(
        args.db, args.limit, args.rebuild, args.threads, args.mem_limit, args.parquet_dir or None
    )


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
    g.add_argument("--scene", default=None, help="genre token(s), comma-separated for exact multi-genre match")
    g.set_defaults(func=cmd_make)

    b = sub.add_parser("build", help="materialize track signals (+ optional co-occurrence affinity) from ingested data")
    b.add_argument("--skip-affinity", action="store_true")
    b.set_defaults(func=cmd_build)

    gg = sub.add_parser("genres", help="import Spotify artist genres from a parquet dump (optional)")
    gg.add_argument("--parquet-dir", default=PARQUET_DIR_DEFAULT or None)
    gg.set_defaults(func=cmd_genres)

    ft = sub.add_parser("features", help="import scalar audio features from a spotify-style parquet dump (optional)")
    ft.add_argument("--parquet-dir", default=PARQUET_DIR_DEFAULT or None)
    ft.add_argument("--limit", type=int, default=None)
    ft.add_argument("--rebuild", action="store_true")
    ft.add_argument("--threads", type=int, default=8)
    ft.add_argument("--mem-limit", default="3GB")
    ft.set_defaults(func=cmd_features)

    args = ap.parse_args(argv)
    if "all" in getattr(args, "sources", []):
        args.sources = ["files", "strawberry", "ipod", "spotify-history"]
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
