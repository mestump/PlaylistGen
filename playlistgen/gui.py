"""Interactive CLI-based text interface using ``questionary``."""

from __future__ import annotations

import logging

import questionary

from .config import load_config, save_config
from .pipeline import run_pipeline
from .seed_playlist import build_seed_playlist


def edit_tokens(cfg: dict) -> None:
    """Prompt user for API tokens and save them."""
    lastfm = questionary.text(
        "Last.fm API key", default=str(cfg.get("LASTFM_API_KEY") or "")
    ).ask()
    cid = questionary.text(
        "Spotify Client ID", default=str(cfg.get("SPOTIFY_CLIENT_ID", ""))
    ).ask()
    secret = questionary.text(
        "Spotify Client Secret", default=str(cfg.get("SPOTIFY_CLIENT_SECRET", ""))
    ).ask()
    anthropic_key = questionary.text(
        "Anthropic API key (for AI naming/enrichment/curation — leave blank to disable)",
        default=str(cfg.get("ANTHROPIC_API_KEY") or ""),
    ).ask()
    history_path = questionary.text(
        "Spotify streaming history path (StreamingHistory*.json or directory — leave blank to skip)",
        default=str(cfg.get("SPOTIFY_HISTORY_PATH") or ""),
    ).ask()
    if lastfm:
        cfg["LASTFM_API_KEY"] = lastfm
    if cid:
        cfg["SPOTIFY_CLIENT_ID"] = cid
    if secret:
        cfg["SPOTIFY_CLIENT_SECRET"] = secret
    if anthropic_key:
        cfg["ANTHROPIC_API_KEY"] = anthropic_key
        cfg["AI_ENHANCE"] = True
    if history_path:
        cfg["SPOTIFY_HISTORY_PATH"] = history_path
    save_config(cfg)


def spotify_login(cfg: dict) -> None:
    """Attempt to get a Spotify OAuth token."""
    try:
        import spotipy
        from spotipy.oauth2 import SpotifyOAuth

        auth = SpotifyOAuth(
            client_id=cfg.get("SPOTIFY_CLIENT_ID"),
            client_secret=cfg.get("SPOTIFY_CLIENT_SECRET"),
            scope="playlist-modify-private",
            redirect_uri=cfg.get("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback"),
        )
        token = auth.get_access_token(as_dict=False)
        cfg["SPOTIFY_TOKEN"] = token
        save_config(cfg)
        logging.info("Spotify authentication successful")
    except Exception:  # pragma: no cover
        logging.exception("Spotify authentication failed")


def edit_config(cfg: dict) -> None:
    """Generic config editor."""
    key = questionary.select("Config option", choices=sorted(cfg.keys())).ask()
    if not key:
        return
    val = questionary.text("New value", default=str(cfg.get(key, ""))).ask()
    cfg[key] = val
    save_config(cfg)


def run_gui() -> str | None:
    """Launch an interactive text user interface and execute the chosen action."""
    cfg = load_config()
    action = questionary.select(
        "Select an action",
        choices=[
            "Generate mix",
            "Generate from seed song",
            "Random mix",
            "Filter by mood",
            "Filter by genre",
            "AI Curate playlists",
            "AI Batch Enrich library",
            "Discover new music",
            "Force recache",
            "Recache moods",
            "Set API tokens",
            "Spotify login",
            "Edit config",
            "Exit",
        ],
    ).ask()

    if action == "Generate mix":
        genre = questionary.text("Genre filter (leave blank for none)").ask()
        mood = questionary.text("Mood filter (leave blank for none)").ask()
        run_pipeline(cfg, genre=genre or None, mood=mood or None)

    elif action == "Generate from seed song":
        song = questionary.text("Seed song 'Artist - Title'").ask()
        num = questionary.text("Number of tracks", default="20").ask()
        build_seed_playlist(song, cfg=cfg, limit=int(num))

    elif action == "Random mix":
        run_pipeline(cfg)

    elif action == "Filter by mood":
        mood = questionary.text("Mood").ask()
        run_pipeline(cfg, mood=mood or None)

    elif action == "Filter by genre":
        genre = questionary.text("Genre").ask()
        run_pipeline(cfg, genre=genre or None)

    elif action == "AI Curate playlists":
        if not cfg.get("ANTHROPIC_API_KEY"):
            print("ANTHROPIC_API_KEY not set. Use 'Set API tokens' first.")
            return action
        cfg_copy = {**cfg, "AI_CURATE": True}
        run_pipeline(cfg_copy)

    elif action == "AI Batch Enrich library":
        if not cfg.get("ANTHROPIC_API_KEY"):
            print("ANTHROPIC_API_KEY not set. Use 'Set API tokens' first.")
            return action
        from .pipeline import ensure_itunes_json
        from .itunes import load_itunes_json
        from .ai_enhancer import batch_enrich_metadata

        itunes_json = ensure_itunes_json(cfg)
        library_df = load_itunes_json(str(itunes_json))
        enrich_cache = cfg.get(
            "AI_ENRICH_CACHE_DB",
            str(__import__("pathlib").Path.home() / ".playlistgen" / "claude_enrichment.sqlite"),
        )
        enriched_df = batch_enrich_metadata(
            library_df,
            api_key=cfg["ANTHROPIC_API_KEY"],
            model=cfg.get("AI_MODEL", "claude-haiku-4-5-20251001"),
            cache_db=enrich_cache,
        )
        mood_count = (enriched_df["Mood"].notna() & (enriched_df["Mood"] != "Unknown")).sum()
        print(f"Enrichment complete: Mood populated for {mood_count} / {len(enriched_df)} tracks.")

    elif action == "Discover new music":
        genre = questionary.text("Genre or search query (e.g. 'Indie Rock')").ask()
        limit = questionary.text("Number of Spotify playlists to scrape", default="5").ask()
        if not genre:
            logging.warning("Genre is required for discovery mode.")
            return action
        from .ai_enhancer import discover_similar
        from .tag_mood_service import load_tag_mood_db
        from .scoring import score_tracks
        from .playlist_builder import save_m3u
        from .pipeline import ensure_itunes_json, ensure_tag_cache
        from .itunes import load_itunes_json

        itunes_json = ensure_itunes_json(cfg)
        ensure_tag_cache(cfg, itunes_json)
        library_df = load_itunes_json(str(itunes_json))
        tag_db = load_tag_mood_db()
        scored_df = score_tracks(library_df, tag_mood_db=tag_db)

        result = discover_similar(
            genre=genre,
            library_df=scored_df,
            cfg=cfg,
            limit=int(limit or 5),
        )
        if result:
            label, playlist_df = result
            save_m3u(playlist_df, label, out_dir=cfg.get("OUTPUT_DIR", "./mixes"))
            print(f"Discovery playlist '{label}' written.")
        else:
            print("Discovery failed. Check Spotify credentials in config.")

    elif action == "Force recache":
        from .pipeline import ensure_itunes_json, ensure_tag_cache
        itunes_json = ensure_itunes_json(cfg)
        ensure_tag_cache(cfg, itunes_json)

    elif action == "Recache moods":
        from .pipeline import ensure_itunes_json, ensure_tag_cache
        itunes_json = ensure_itunes_json(cfg)
        ensure_tag_cache(cfg, itunes_json)

    elif action == "Set API tokens":
        edit_tokens(cfg)

    elif action == "Spotify login":
        spotify_login(cfg)

    elif action == "Edit config":
        edit_config(cfg)

    return action
