"""
Main orchestration pipeline for PlaylistGen.

Wires together all stages:
  1. Library loading   (iTunes XML or local directory + mutagen enrichment)
  2. Last.fm tag cache (SQLite, rate-limited, resume-friendly)
  3. Taste profile     (from Spotify history — optional)
  4. Scoring           (genre, mood, year, play/skip counts — all bugs fixed)
  5. Clustering        (mood / year / KMeans)
  6. AI enhancement    (Claude API playlist naming — optional)
  7. Playlist building (energy-arc ordering, M3U export)
  8. Feedback          (record 'generated' event per playlist)
"""

import logging
import random
from pathlib import Path

from .config import load_config
from .itunes import convert_itunes_xml, load_itunes_json, build_library_from_dir, save_itunes_json
from .tag_mood_service import generate_tag_mood_cache, load_tag_mood_db
from .spotify_profile import build_profile, load_profile
from .scoring import score_tracks
from .clustering import cluster_tracks, name_cluster, humanize_label
from .playlist_builder import build_playlists
from .feedback import load_feedback, save_feedback, update_feedback
from .mood_map import build_tag_counts


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------


def ensure_itunes_json(cfg: dict) -> Path:
    """
    Convert iTunes XML → JSON if the JSON is missing or older than the XML.
    Returns the path to the (now-current) JSON file.
    """
    itunes_json = Path(cfg["ITUNES_JSON"])
    itunes_xml = Path(cfg.get("ITUNES_XML", "iTunes Music Library.xml"))
    if not itunes_json.exists() or (
        itunes_xml.exists()
        and itunes_xml.stat().st_mtime > itunes_json.stat().st_mtime
    ):
        logging.info("Converting iTunes XML → JSON: %s → %s", itunes_xml, itunes_json)
        convert_itunes_xml(str(itunes_xml), str(itunes_json))
    return itunes_json


def ensure_tag_cache(cfg: dict, itunes_json: Path) -> None:
    """
    Fetch Last.fm tags for all tracks in the library + Spotify history.
    Skips tracks already cached in SQLite (resume-friendly).
    Does nothing if LASTFM_API_KEY is not set.
    """
    if not cfg.get("LASTFM_API_KEY"):
        logging.info(
            "LASTFM_API_KEY not set — skipping tag enrichment. "
            "Mood detection will rely on iTunes/embedded Genre tags only."
        )
        return
    generate_tag_mood_cache(
        itunes_json_path=str(itunes_json),
        spotify_dir=cfg.get("SPOTIFY_DIR"),
        tag_mood_path=cfg.get("TAG_MOOD_CACHE"),
    )


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------


def run_pipeline(
    cfg: dict = None,
    genre: str = None,
    mood: str = None,
    library_dir: str = None,
    no_ai: bool = False,
) -> list:
    """
    Generate playlists from the user's music library.

    Args:
        cfg:         Config dict (loaded from config.yml if None).
        genre:       If set, build a single playlist filtered to this genre.
        mood:        If set, build a single playlist filtered to this mood.
        library_dir: Scan a local directory instead of using iTunes XML.
        no_ai:       Suppress AI enhancement even if AI_ENHANCE=true in config.

    Returns:
        List of (label, DataFrame) tuples for each playlist built.
    """
    logging.basicConfig(level=logging.INFO)
    if cfg is None:
        cfg = load_config()

    logging.info("=== PlaylistGen pipeline starting ===")

    # ------------------------------------------------------------------
    # Stage 1: Library loading
    # ------------------------------------------------------------------
    if library_dir:
        logging.info("Scanning local library: %s", library_dir)
        df = build_library_from_dir(library_dir)
        if df.empty:
            logging.error("No audio files found in %s — aborting.", library_dir)
            return []
        itunes_json = Path(cfg["ITUNES_JSON"])
        save_itunes_json(df, itunes_json)
    else:
        itunes_json = ensure_itunes_json(cfg)
        df = load_itunes_json(str(itunes_json))
        if df.empty:
            logging.error("Library is empty — aborting.")
            return []

    logging.info("Library loaded: %d tracks.", len(df))

    # ------------------------------------------------------------------
    # Stage 2: Last.fm tag cache
    # ------------------------------------------------------------------
    ensure_tag_cache(cfg, itunes_json)
    tag_db = load_tag_mood_db()
    tag_counts = build_tag_counts(tag_db)
    logging.info("Tag DB loaded: %d entries.", len(tag_db))

    # ------------------------------------------------------------------
    # Stage 3: Taste profile (Spotify history — optional)
    # ------------------------------------------------------------------
    spotify_dir = Path(cfg.get("SPOTIFY_DIR", "./spotify_history"))
    profile_path = Path(cfg.get("PROFILE_PATH", "./taste_profile.json"))

    if spotify_dir.exists() and any(spotify_dir.rglob("*.json")):
        try:
            profile = build_profile(
                spotify_dir=str(spotify_dir),
                out_path=str(profile_path),
                tag_db=tag_db,
            )
        except Exception as exc:
            logging.warning("Profile build failed: %s — using empty profile.", exc)
            profile = {}
    else:
        logging.info(
            "No Spotify history at %s — personalization disabled.", spotify_dir
        )
        profile = {}

    # ------------------------------------------------------------------
    # Stage 4: Scoring
    # ------------------------------------------------------------------
    logging.info("Scoring tracks…")
    scored_df = score_tracks(df, config=profile, tag_mood_db=tag_db)

    # ------------------------------------------------------------------
    # Stage 4b: Genre / Mood filter (single playlist mode)
    # ------------------------------------------------------------------
    if genre or mood:
        filt = scored_df.copy()
        if genre:
            filt = filt[
                filt["Genre"].notna()
                & (filt["Genre"].str.lower() == genre.lower())
            ]
        if mood:
            filt = filt[
                filt["Mood"].notna()
                & (filt["Mood"].str.lower() == mood.lower())
            ]
        if filt.empty:
            logging.warning(
                "No tracks match genre=%r mood=%r. Try broader filters.", genre, mood
            )
            return []
        label = humanize_label(mood, genre)
        return build_playlists([filt], scored_df, name_fn=lambda *_: label)

    # ------------------------------------------------------------------
    # Stage 5: Clustering
    # ------------------------------------------------------------------
    n_clusters = int(cfg.get("CLUSTER_COUNT", 6))
    num_playlists = int(cfg.get("NUM_PLAYLISTS", n_clusters))
    cluster_by_year = bool(cfg.get("YEAR_MIX_ENABLED", False))
    year_range = int(cfg.get("YEAR_MIX_RANGE", 0))
    cluster_by_mood = bool(cfg.get("CLUSTER_BY_MOOD", False))
    min_tracks_per_year = int(cfg.get("MIN_TRACKS_PER_YEAR", 10))

    clusters = cluster_tracks(
        scored_df,
        n_clusters=n_clusters,
        cluster_by_year=cluster_by_year,
        year_range=year_range,
        cluster_by_mood=cluster_by_mood,
        min_tracks_per_year=min_tracks_per_year,
    )

    # Shuffle and cap
    random.shuffle(clusters)
    selected = clusters[:num_playlists]

    # Build preliminary labels (used for AI input and fallback)
    labelled = [(name_cluster(cl, i), cl) for i, cl in enumerate(selected)]

    # ------------------------------------------------------------------
    # Stage 6: AI enhancement (optional)
    # ------------------------------------------------------------------
    ai_enabled = bool(cfg.get("AI_ENHANCE", False)) and not no_ai
    if ai_enabled:
        api_key = cfg.get("ANTHROPIC_API_KEY")
        if api_key:
            try:
                from .ai_enhancer import enhance_playlists
                labelled = enhance_playlists(
                    labelled,
                    api_key=api_key,
                    model=cfg.get("AI_MODEL", "claude-haiku-4-5-20251001"),
                )
            except Exception as exc:
                logging.warning("AI enhancement failed: %s — using generated labels.", exc)
        else:
            logging.info("AI_ENHANCE=true but ANTHROPIC_API_KEY not set — skipping.")

    # ------------------------------------------------------------------
    # Stage 7: Playlist building + M3U export
    # ------------------------------------------------------------------
    playlists = build_playlists(
        [cl for _, cl in labelled],
        scored_df,
        num_playlists=num_playlists,
        name_fn=lambda cl, i: labelled[i][0],
    )

    # ------------------------------------------------------------------
    # Stage 8: Feedback
    # ------------------------------------------------------------------
    feedback_path = Path(cfg.get("FEEDBACK_PATH", Path.home() / ".playlistgen" / "feedback.json"))
    for label, _ in playlists:
        update_feedback(str(feedback_path), label, "generated")

    logging.info("=== Pipeline complete. %d playlists written to %s ===",
                 len(playlists), cfg.get("OUTPUT_DIR", "./mixes"))
    return playlists


# ---------------------------------------------------------------------------
# Convenience re-exports used by cli.py
# ---------------------------------------------------------------------------


def ensure_tag_mood_cache(cfg: dict, itunes_json: Path) -> Path:
    """Backward-compat shim for cli.py recache-moods command."""
    ensure_tag_cache(cfg, itunes_json)
    return Path(cfg.get("TAG_MOOD_CACHE", Path.home() / ".playlistgen" / "lastfm_tags_cache.json"))
