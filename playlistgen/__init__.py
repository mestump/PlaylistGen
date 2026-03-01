"""PlaylistGen package."""

from .pipeline import run_pipeline
from .itunes import load_itunes_json, build_library_from_dir, convert_itunes_xml
from .scoring import score_tracks
from .clustering import cluster_tracks, name_cluster, humanize_label
from .playlist_builder import build_playlists, save_m3u
from .spotify_profile import build_profile, load_profile
from .mood_map import canonical_mood, canonical_genre, build_tag_counts
from .lastfm_client import load_tag_db_from_sqlite, generate_tag_cache
from .metadata import read_audio_tags, enrich_dataframe
from .feedback import load_feedback, save_feedback, update_feedback
from .seed_playlist import build_seed_playlist
from .pattern_analyzer import analyze_playlists, vectorize_playlists, vectorize_playlist
from .similarity import score_playlists
from .train_model import train_cluster_model
from .playlist_scraper import (
    fetch_spotify_playlists,
    fetch_youtube_playlists,
    fetch_apple_music_playlists,
)

try:
    from .gui import run_gui
except ImportError:
    run_gui = None  # questionary not installed

__all__ = [
    "run_pipeline",
    "load_itunes_json",
    "build_library_from_dir",
    "convert_itunes_xml",
    "score_tracks",
    "cluster_tracks",
    "name_cluster",
    "humanize_label",
    "build_playlists",
    "save_m3u",
    "build_profile",
    "load_profile",
    "canonical_mood",
    "canonical_genre",
    "build_tag_counts",
    "load_tag_db_from_sqlite",
    "generate_tag_cache",
    "read_audio_tags",
    "enrich_dataframe",
    "load_feedback",
    "save_feedback",
    "update_feedback",
    "build_seed_playlist",
    "analyze_playlists",
    "vectorize_playlists",
    "vectorize_playlist",
    "score_playlists",
    "train_cluster_model",
    "fetch_spotify_playlists",
    "fetch_youtube_playlists",
    "fetch_apple_music_playlists",
    "run_gui",
]
