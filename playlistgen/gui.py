"""Interactive CLI-based GUI using questionary."""

import questionary

from .config import load_config
from .pipeline import run_pipeline
from .seed_playlist import build_seed_playlist


def run_gui():
    """Launch an interactive text UI."""
    cfg = load_config()
    action = questionary.select(
        "Select an action",
        choices=[
            "Generate mix",
            "Generate from seed song",
            "Recache moods",
            "Exit",
        ],
    ).ask()

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
    return action
