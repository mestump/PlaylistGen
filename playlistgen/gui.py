"""Interactive CLI-based GUI using questionary."""


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
    if lastfm:
        cfg["LASTFM_API_KEY"] = lastfm
    if cid:
        cfg["SPOTIFY_CLIENT_ID"] = cid
    if secret:
        cfg["SPOTIFY_CLIENT_SECRET"] = secret
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
            redirect_uri="http://localhost:8888/callback",
        )
        token = auth.get_access_token(as_dict=False)
        cfg["SPOTIFY_TOKEN"] = token
        save_config(cfg)
        logging.info("Spotify authentication successful")
    except Exception:  # pragma: no cover - network or spotipy issues
        logging.exception("Spotify authentication failed")


def edit_config(cfg: dict) -> None:
    """Generic config editor."""
    key = questionary.select("Config option", choices=sorted(cfg.keys())).ask()
    if not key:
        return
    val = questionary.text("New value", default=str(cfg.get(key, ""))).ask()
    cfg[key] = val
    save_config(cfg)


def run_gui():
    """Launch an interactive text UI."""
    cfg = load_config()
    action = questionary.select(
        "Select an action",
        choices=[

            "Random mix",
            "Filter by mood",
            "Filter by genre",
            "Seed from song",
            "Force recache",
            "Set API tokens",
            "Spotify login",
            "Edit config",
            "Generate mix",
            "Generate from seed song",
            "Recache moods",

            "Exit",
        ],
    ).ask()


    if action == "Random mix":
        run_pipeline(cfg)
    elif action == "Filter by mood":
        mood = questionary.text("Mood").ask()
        run_pipeline(cfg, mood=mood or None)
    elif action == "Filter by genre":
        genre = questionary.text("Genre").ask()
        run_pipeline(cfg, genre=genre or None)
    elif action == "Seed from song":
        song = questionary.text("Seed song 'Artist - Title'").ask()
        num = questionary.text("Number of tracks", default="20").ask()
        build_seed_playlist(song, cfg=cfg, limit=int(num))
    elif action == "Force recache":

    if action == "Generate mix":
        genre = questionary.text(
            "Genre filter (leave blank for none)"
        ).ask()
        mood = questionary.text(
            "Mood filter (leave blank for none)"
        ).ask()
        run_pipeline(cfg, genre=genre or None, mood=mood or None)
    elif action == "Generate from seed song":
        song = questionary.text("Seed song 'Artist - Title'").ask()
        num = questionary.text("Number of tracks", default="20").ask()
        build_seed_playlist(song, cfg=cfg, limit=int(num))
    elif action == "Recache moods":

        from .pipeline import ensure_itunes_json, ensure_tag_mood_cache

        itunes_json = ensure_itunes_json(cfg)
        ensure_tag_mood_cache(cfg, itunes_json)

    elif action == "Set API tokens":
        edit_tokens(cfg)
    elif action == "Spotify login":
        spotify_login(cfg)
    elif action == "Edit config":
        edit_config(cfg)
=======

    return action
