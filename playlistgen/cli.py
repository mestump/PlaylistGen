"""CLI"""

import argparse
import logging
import os
from pathlib import Path


from .config import load_config
from .pipeline import run_pipeline


def file_newer(a, b):
    """Return True if file `a` exists and is newer than file `b` (or if `b` is missing)."""
    if not os.path.exists(a):
        return False
    if not os.path.exists(b):
        return True
    return os.path.getmtime(a) > os.path.getmtime(b)


def main():
    parser = argparse.ArgumentParser(prog="playlistgen", description="PlaylistGen CLI")
    parser.add_argument(
        "--log-level", help="Override log level (DEBUG, INFO, WARNING, ERROR)"
    )
    parser.add_argument(
        "--genre", help="Filter mix to only tracks matching the given genre"
    )
    parser.add_argument(
        "--mood", help="Filter mix to only tracks matching the given mood"
    )
    parser.add_argument(
        "--library-dir",
        help="Path to a manual music library directory (used if no iTunes library)",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("recache-moods", help="Force re-cache all moods from Last.fm")

    subparsers.add_parser("gui", help="Launch interactive TUI")

    seed_parser = subparsers.add_parser(
        "seed-song", help="Generate a playlist from a seed song"
    )
    seed_parser.add_argument("--song", required=True, help="Seed song as 'Artist - Title'")
    seed_parser.add_argument("--num", type=int, default=20, help="Number of tracks in the mix")

    args = parser.parse_args()
    cfg = load_config()

    lvl = args.log_level or cfg.get("LOG_LEVEL") or "INFO"
    logging.basicConfig(level=getattr(logging, lvl.upper(), logging.INFO))

    if args.command == "recache-moods":
        from .pipeline import ensure_itunes_json, ensure_tag_mood_cache

        itunes_json = ensure_itunes_json(cfg)
        tag_path = Path(cfg["TAG_MOOD_CACHE"])
        cache_db = Path(cfg.get("CACHE_DB"))
        if tag_path.exists():
            tag_path.unlink()
        if cache_db.exists():
            cache_db.unlink()

        ensure_tag_mood_cache(cfg, itunes_json)
    elif args.command == "gui":
        from .gui import run_gui

        run_gui()
    elif args.command == "seed-song":
        from .seed_playlist import build_seed_playlist

        build_seed_playlist(
            args.song, cfg=cfg, library_dir=args.library_dir, limit=args.num
        )
    else:
        run_pipeline(
            cfg, genre=args.genre, mood=args.mood, library_dir=args.library_dir
        )


if __name__ == "__main__":
    main()
