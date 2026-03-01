"""CLI entry point for PlaylistGen."""

import argparse
import logging
import os
from pathlib import Path

from .config import load_config
from .pipeline import run_pipeline


def file_newer(a, b):
    """Return True if file `a` exists and is newer than file `b` (or `b` is missing)."""
    if not os.path.exists(a):
        return False
    if not os.path.exists(b):
        return True
    return os.path.getmtime(a) > os.path.getmtime(b)


def main():
    parser = argparse.ArgumentParser(
        prog="playlistgen",
        description="PlaylistGen — Spotify-quality playlists from your local library",
    )
    parser.add_argument("--log-level", help="Log level: DEBUG, INFO, WARNING, ERROR")
    parser.add_argument("--genre", help="Filter to tracks matching this genre")
    parser.add_argument("--mood", help="Filter to tracks matching this mood")
    parser.add_argument(
        "--library-dir",
        help="Scan a local music directory (bypasses iTunes XML)",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Disable Claude AI playlist naming even if AI_ENHANCE=true in config",
    )

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "recache-moods",
        help="Force re-fetch Last.fm tags for the whole library",
    )

    subparsers.add_parser("gui", help="Launch the interactive text UI")

    seed_parser = subparsers.add_parser(
        "seed-song", help="Build a playlist from a seed track via Last.fm similarity"
    )
    seed_parser.add_argument(
        "--song", required=True, help="Seed song in 'Artist - Title' format"
    )
    seed_parser.add_argument(
        "--num", type=int, default=20, help="Number of tracks in the playlist"
    )

    discover_parser = subparsers.add_parser(
        "discover",
        help=(
            "Discover new music by scraping similar Spotify playlists and "
            "finding matches in your library"
        ),
    )
    discover_parser.add_argument(
        "--genre",
        required=True,
        help="Genre or search query (e.g. 'Indie Rock')",
    )
    discover_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of Spotify playlists to scrape (default: 5)",
    )

    args = parser.parse_args()
    cfg = load_config()

    lvl = args.log_level or cfg.get("LOG_LEVEL") or "INFO"
    logging.basicConfig(level=getattr(logging, lvl.upper(), logging.INFO))

    # ------------------------------------------------------------------
    # Command routing
    # ------------------------------------------------------------------
    if args.command == "recache-moods":
        from .pipeline import ensure_itunes_json, ensure_tag_cache

        itunes_json = ensure_itunes_json(cfg)

        # Remove SQLite cache to force a full re-fetch
        db_path = (
            cfg.get("LASTFM_CACHE_DB")
            or cfg.get("CACHE_DB")
            or str(Path.home() / ".playlistgen" / "lastfm.sqlite")
        )
        if Path(db_path).exists():
            Path(db_path).unlink()
            logging.info("Removed tag cache at %s — will re-fetch on next run.", db_path)

        ensure_tag_cache(cfg, itunes_json)

    elif args.command == "gui":
        from .gui import run_gui
        run_gui()

    elif args.command == "seed-song":
        from .seed_playlist import build_seed_playlist
        build_seed_playlist(
            args.song, cfg=cfg, library_dir=getattr(args, "library_dir", None), limit=args.num
        )

    elif args.command == "discover":
        from .ai_enhancer import discover_similar
        from .pipeline import ensure_itunes_json, ensure_tag_cache
        from .tag_mood_service import load_tag_mood_db
        from .scoring import score_tracks
        from .playlist_builder import save_m3u

        lib_dir = getattr(args, "library_dir", None)
        if lib_dir:
            from .itunes import build_library_from_dir
            library_df = build_library_from_dir(lib_dir)
        else:
            itunes_json = ensure_itunes_json(cfg)
            ensure_tag_cache(cfg, itunes_json)
            from .itunes import load_itunes_json
            library_df = load_itunes_json(str(itunes_json))

        tag_db = load_tag_mood_db()
        scored_df = score_tracks(library_df, tag_mood_db=tag_db)

        result = discover_similar(
            genre=args.genre,
            library_df=scored_df,
            cfg=cfg,
            limit=args.limit,
        )
        if result:
            label, playlist_df = result
            save_m3u(playlist_df, label, out_dir=cfg.get("OUTPUT_DIR", "./mixes"))
            logging.info("Discovery playlist '%s' written.", label)
        else:
            logging.error(
                "Discovery failed. Ensure SPOTIFY_CLIENT_ID and "
                "SPOTIFY_CLIENT_SECRET are set in config.yml."
            )

    elif args.command is None and (args.genre or args.mood or args.library_dir):
        run_pipeline(
            cfg,
            genre=args.genre,
            mood=args.mood,
            library_dir=args.library_dir,
            no_ai=getattr(args, "no_ai", False),
        )

    else:
        from .gui import run_gui
        run_gui()


if __name__ == "__main__":
    main()
