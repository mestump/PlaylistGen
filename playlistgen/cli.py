"""CLI"""

import argparse
import logging
import os
from pathlib import Path


from .config import load_config
from .pipeline import run_pipeline
from .local_scanner import scan_local_library


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
        '--library-path',
        type=str,
        help='Path to the root of your local music library. If provided, iTunes library will be ignored.',
        default=None 
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

    # This top-level import of ensure_tag_mood_cache assumes it will be correctly refactored in pipeline.py
    from .pipeline import ensure_tag_mood_cache 
    import pandas as pd

    track_data_for_pipeline = None
    source_description = "iTunes library"

    if args.library_path:
        logging.info(f"Scanning local library at: {args.library_path}")
        track_data_for_pipeline = scan_local_library(args.library_path)
        source_description = "local library scan"
    
    if args.command == 'recache-moods':
        tag_path = Path(cfg['TAG_MOOD_CACHE'])
        if tag_path.exists():
            tag_path.unlink(missing_ok=True)
        
        current_tracks_df = None
        if track_data_for_pipeline: # Data from local scan
            logging.info(f"Recaching moods based on {source_description}.")
            current_tracks_df = pd.DataFrame(track_data_for_pipeline)
        else: # Fallback to iTunes
            logging.info("Recaching moods based on iTunes library.")
            from .pipeline import ensure_itunes_json # For fallback
            from .itunes import load_itunes_json    # For fallback
            itunes_json_path = ensure_itunes_json(cfg) # ensure_itunes_json is still relevant for this path
            current_tracks_df = load_itunes_json(str(itunes_json_path))
        
        if current_tracks_df is not None and not current_tracks_df.empty:
            ensure_tag_mood_cache(cfg, current_tracks_df) # Call refactored version
            logging.info("Mood cache recached.")
        else:
            logging.warning("No track data found to recache moods.")

    elif args.command is None: # Regular pipeline run
        if track_data_for_pipeline:
            logging.info(f"Running pipeline with data from {source_description}.")
            # local_track_data is passed as a list of dicts
            run_pipeline(cfg, genre=args.genre, mood=args.mood, local_track_data=track_data_for_pipeline)
        else:
            logging.info(f"Running pipeline with data from {source_description}.")
            run_pipeline(cfg, genre=args.genre, mood=args.mood) # local_track_data remains None
    # else:
        # Future commands can be handled here

if __name__ == "__main__":
    main()
