"""PlaylistGen package."""

from .playlist_scraper import (
    fetch_spotify_playlists,
    fetch_youtube_playlists,
    fetch_apple_music_playlists,
)
from .pattern_analyzer import (
    analyze_playlists,
    vectorize_playlists,
    vectorize_playlist,
)
from .playlist_generator import generate_candidates
from .similarity import score_playlists
from .feedback import load_feedback, save_feedback, update_feedback

__all__ = [
    "fetch_spotify_playlists",
    "fetch_youtube_playlists",
    "fetch_apple_music_playlists",
    "analyze_playlists",
    "vectorize_playlists",
    "vectorize_playlist",
    "generate_candidates",
    "score_playlists",
    "load_feedback",
    "save_feedback",
    "update_feedback",
]
