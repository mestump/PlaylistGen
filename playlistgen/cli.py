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
    parser.add_argument(
        '--library-input-path',
        help='Path to your music library (directory, M3U file, or iTunes XML file). Overrides LIBRARY_INPUT_PATH in config.',
        type=str,
        default=None 
    )
    subparsers = parser.add_subparsers(dest='command')

    subparsers.add_parser(
        'recache-moods',
        help='Force re-cache all moods from Last.fm'
    )

    args = parser.parse_args()
    cfg = load_config()

    # Override config with CLI arguments if provided
    if args.library_input_path:
        cfg['LIBRARY_INPUT_PATH'] = args.library_input_path
        # Use logging.debug or logging.info as appropriate after logging is configured
        # logging.info(f"Overriding LIBRARY_INPUT_PATH with CLI argument: {args.library_input_path}")

    lvl = args.log_level or cfg.get('LOG_LEVEL') or 'INFO'
    logging.basicConfig(level=getattr(logging, lvl.upper(), logging.INFO))
    
    # Log the override after basicConfig is set up so it's visible
    if args.library_input_path:
        logging.info(f"Using library input path from command line: {args.library_input_path}")


    if args.command == 'recache-moods':
        # ensure_itunes_json is now ensure_library_json in the new pipeline
        # For recache-moods, we need to ensure the library JSON exists to derive tags.
        # The new pipeline.py is not used yet, so this part will eventually need adjustment
        # if pipeline.py is swapped. For now, assuming old pipeline.py for this specific command.
        from .pipeline import ensure_itunes_json, ensure_tag_mood_cache # Assuming old names for now for this block

        itunes_json = ensure_itunes_json(cfg) # This will use ITUNES_XML or LIBRARY_INPUT_PATH if old ensure_itunes_json is already updated
        tag_path = Path(cfg['TAG_MOOD_CACHE'])
        cache_db = Path(cfg.get('CACHE_DB')) # This key might not exist or be relevant anymore
        if tag_path.exists():
            tag_path.unlink()
        if cache_db.exists():
            cache_db.unlink()

        ensure_tag_mood_cache(cfg, itunes_json)
    else:
        run_pipeline(cfg, genre=args.genre, mood=args.mood)

if __name__ == "__main__":
    main()
