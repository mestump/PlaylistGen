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
    parser = argparse.ArgumentParser(prog='playlistgen', description='PlaylistGen CLI')
    parser.add_argument(
        '--log-level',
        help='Override log level (DEBUG, INFO, WARNING, ERROR)'
    )
    parser.add_argument(
        '--genre',
        help='Filter mix to only tracks matching the given genre'
    )
    parser.add_argument(
        '--mood',
        help='Filter mix to only tracks matching the given mood'
    )
    subparsers = parser.add_subparsers(dest='command')

    subparsers.add_parser(
        'recache-moods',
        help='Force re-cache all moods from Last.fm'
    )

    args = parser.parse_args()
    cfg = load_config()

    lvl = args.log_level or cfg.get('LOG_LEVEL') or 'INFO'
    logging.basicConfig(level=getattr(logging, lvl.upper(), logging.INFO))

    if args.command == 'recache-moods':
        from .pipeline import ensure_itunes_json, ensure_tag_mood_cache

        itunes_json = ensure_itunes_json(cfg)
        tag_path = Path(cfg['TAG_MOOD_CACHE'])
        cache_db = Path(cfg.get('CACHE_DB'))
        if tag_path.exists():
            tag_path.unlink()
        if cache_db.exists():
            cache_db.unlink()

        ensure_tag_mood_cache(cfg, itunes_json)
    else:
        run_pipeline(cfg, genre=args.genre, mood=args.mood)

if __name__ == "__main__":
    main()
