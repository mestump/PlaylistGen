"""
Main orchestration pipeline for PlaylistGen.

Wires together all stages:
  1. Library loading      (iTunes XML or local directory + mutagen enrichment)
  2. Audio analysis       (libROSA — local BPM/energy/spectral; SQLite cached)
  3. Metadata enrichment  (Claude batch | Last.fm | embedded genre — priority chain)
  4. Session model        (co-occurrence + recency from Spotify streaming JSON)
  5. Taste profile        (from Spotify history — optional)
  6. Scoring              (genre, mood, year, play/skip + recency + co-occurrence)
  7. Clustering / Curation (audio features | mood | tfidf | Claude AI curation)
  8. AI naming            (Claude Haiku playlist naming — optional)
  9. Playlist building    (energy-arc ordering, M3U export)
  10. Feedback            (record 'generated' event per playlist)
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
            "LASTFM_API_KEY not set — skipping Last.fm tag enrichment. "
            "Mood detection will use Claude batch enrichment or embedded Genre tags."
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
        no_ai:       Suppress all AI features even if enabled in config.

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
        mutagen_enabled = bool(cfg.get("MUTAGEN_ENABLED", True))
        df = build_library_from_dir(library_dir, mutagen_enabled=mutagen_enabled)
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
    # Stage 2: Local audio analysis (libROSA — optional, SQLite cached)
    # ------------------------------------------------------------------
    librosa_enabled = bool(cfg.get("LIBROSA_ENABLED", True))
    if librosa_enabled:
        try:
            from .audio_analysis import analyze_library

            audio_cache = str(
                Path(
                    cfg.get(
                        "AUDIO_CACHE_DB",
                        Path.home() / ".playlistgen" / "audio.sqlite",
                    )
                ).expanduser()
            )
            workers = int(cfg.get("AUDIO_ANALYSIS_WORKERS", 0))
            duration = int(cfg.get("AUDIO_ANALYSIS_DURATION", 120))
            df = analyze_library(
                df,
                db_path=audio_cache,
                enabled=librosa_enabled,
                workers=workers,
                duration=duration,
            )
        except Exception as exc:
            logging.warning("Audio analysis stage failed: %s — continuing.", exc)

    # ------------------------------------------------------------------
    # Stage 3: Metadata enrichment — priority chain
    #   1. Claude batch enrichment (if AI_BATCH_ENRICH=true + API key set)
    #   2. Last.fm (if LASTFM_API_KEY set)
    #   3. Embedded genre fallback (always available via mood_map)
    # ------------------------------------------------------------------
    ai_batch_enrich = bool(cfg.get("AI_BATCH_ENRICH", False)) and not no_ai
    api_key = cfg.get("ANTHROPIC_API_KEY")

    if ai_batch_enrich and api_key:
        logging.info("Stage 3a: Claude batch metadata enrichment…")
        try:
            from .ai_enhancer import batch_enrich_metadata

            enrich_cache = str(
                Path(
                    cfg.get(
                        "AI_ENRICH_CACHE_DB",
                        Path.home() / ".playlistgen" / "claude_enrichment.sqlite",
                    )
                ).expanduser()
            )
            df = batch_enrich_metadata(
                df,
                api_key=api_key,
                model=cfg.get("AI_MODEL", "claude-haiku-4-5-20251001"),
                cache_db=enrich_cache,
                batch_size=int(cfg.get("AI_ENRICH_BATCH_SIZE", 150)),
                rate_limit_ms=int(cfg.get("AI_ENRICH_RATE_LIMIT_MS", 0)),
            )
        except Exception as exc:
            logging.warning("Claude batch enrichment failed: %s — falling back.", exc)

    # Ollama enrichment fallback: if no Claude API key but Ollama is configured
    ollama_base_url = cfg.get("OLLAMA_BASE_URL")
    if ai_batch_enrich and not api_key and ollama_base_url and not no_ai:
        logging.info("Stage 3a: Ollama batch metadata enrichment (local)…")
        try:
            from .enrichers.ollama_enricher import batch_enrich_ollama

            df = batch_enrich_ollama(
                df,
                base_url=ollama_base_url,
                model=cfg.get("OLLAMA_ENRICH_MODEL", cfg.get("OLLAMA_MODEL", "llama3")),
                batch_size=int(cfg.get("AI_ENRICH_BATCH_SIZE", 50)),
                rate_limit_ms=int(cfg.get("AI_ENRICH_RATE_LIMIT_MS", 0)),
            )
        except Exception as exc:
            logging.warning("Ollama batch enrichment failed: %s — falling back.", exc)

    # Last.fm tag cache (runs even alongside Claude enrichment for tracks Claude missed)
    ensure_tag_cache(cfg, itunes_json)
    tag_db = load_tag_mood_db()
    tag_counts = build_tag_counts(tag_db)
    logging.info("Tag DB loaded: %d entries.", len(tag_db))

    # ------------------------------------------------------------------
    # Stage 4: Session model (Spotify streaming history — optional)
    # ------------------------------------------------------------------
    session_model = None
    history_path = cfg.get("SPOTIFY_HISTORY_PATH")
    if history_path:
        logging.info("Stage 4: Loading session model from %s…", history_path)
        try:
            from .session_model import build_session_model

            session_model = build_session_model(
                history_path,
                gap_minutes=int(cfg.get("SESSION_GAP_MINUTES", 30)),
                half_life_days=int(cfg.get("RECENCY_HALF_LIFE_DAYS", 90)),
            )
        except Exception as exc:
            logging.warning("Session model build failed: %s — continuing.", exc)

    # ------------------------------------------------------------------
    # Stage 5: Taste profile (Spotify listening history — optional)
    # ------------------------------------------------------------------
    spotify_dir = Path(
        cfg.get("SPOTIFY_DIR")
        or cfg.get("SPOTIFY_HISTORY_PATH")
        or "./spotify_history"
    )
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
    # Stage 6: Scoring
    # ------------------------------------------------------------------
    logging.info("Scoring tracks…")
    scored_df = score_tracks(
        df,
        config=profile,
        tag_mood_db=tag_db,
        session_model=session_model,
    )

    # ------------------------------------------------------------------
    # Stage 6b: Genre / Mood filter (single playlist mode)
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
    # Stage 7: Clustering / AI Curation
    # ------------------------------------------------------------------
    n_clusters = int(cfg.get("CLUSTER_COUNT", 6))
    num_playlists = int(cfg.get("NUM_PLAYLISTS", n_clusters))
    cluster_by_year = bool(cfg.get("YEAR_MIX_ENABLED", False))
    year_range = int(cfg.get("YEAR_MIX_RANGE", 0))
    cluster_by_mood = bool(cfg.get("CLUSTER_BY_MOOD", False))
    cluster_hybrid_mode = bool(cfg.get("CLUSTER_HYBRID", False))
    cluster_strategy = cfg.get("CLUSTER_STRATEGY", "auto")
    min_tracks_per_year = int(cfg.get("MIN_TRACKS_PER_YEAR", 10))

    ai_curate = bool(cfg.get("AI_CURATE", False)) and not no_ai
    labelled = None  # set by AI curation or algorithmic clustering

    if ai_curate and api_key:
        logging.info("Stage 7: Claude AI playlist curation…")
        try:
            from .ai_enhancer import claude_curate_playlists

            labelled = claude_curate_playlists(
                scored_df,
                n_playlists=num_playlists,
                api_key=api_key,
                model=cfg.get("AI_CURATE_MODEL", "claude-sonnet-4-6"),
            )
            if not labelled:
                logging.warning(
                    "Claude curation returned no playlists — falling back to clustering."
                )
                labelled = None
        except Exception as exc:
            logging.warning(
                "Claude curation failed: %s — falling back to clustering.", exc
            )
            labelled = None
    elif ai_curate:
        logging.info("AI_CURATE=true but ANTHROPIC_API_KEY not set — using clustering.")

    if labelled is None:
        clusters = cluster_tracks(
            scored_df,
            n_clusters=n_clusters,
            cluster_by_year=cluster_by_year,
            year_range=year_range,
            cluster_by_mood=cluster_by_mood,
            cluster_hybrid_mode=cluster_hybrid_mode,
            min_tracks_per_year=min_tracks_per_year,
            strategy=cluster_strategy,
        )
        # Mood strategy produces exactly one cluster per mood — don't cap.
        # For other strategies, respect num_playlists.
        effective_strategy = cluster_strategy
        if effective_strategy == "auto":
            mood_cov = (
                scored_df["Mood"].notna() & (scored_df["Mood"] != "Unknown")
            ).mean() if "Mood" in scored_df.columns else 0.0
            effective_strategy = "mood" if mood_cov > 0.5 else cluster_strategy
        if effective_strategy == "mood" or cluster_by_mood:
            random.shuffle(clusters)
            selected = clusters
        else:
            random.shuffle(clusters)
            selected = clusters[:num_playlists]
        labelled = [(name_cluster(cl, i), cl) for i, cl in enumerate(selected)]

    # ------------------------------------------------------------------
    # Stage 8: AI naming (when not using AI_CURATE; optional)
    # ------------------------------------------------------------------
    ai_enabled = bool(cfg.get("AI_ENHANCE", False)) and not no_ai and not ai_curate
    if ai_enabled and api_key:
        try:
            from .ai_enhancer import enhance_playlists

            labelled = enhance_playlists(
                labelled,
                api_key=api_key,
                model=cfg.get("AI_MODEL", "claude-haiku-4-5-20251001"),
            )
        except Exception as exc:
            logging.warning("AI naming failed: %s — using generated labels.", exc)
    elif ai_enabled:
        logging.info("AI_ENHANCE=true but ANTHROPIC_API_KEY not set — skipping.")

    # ------------------------------------------------------------------
    # Stage 9: Playlist building + M3U export
    # ------------------------------------------------------------------
    playlists = build_playlists(
        [cl for _, cl in labelled],
        scored_df,
        num_playlists=len(labelled),
        name_fn=lambda cl, i: labelled[i][0],
    )

    # ------------------------------------------------------------------
    # Stage 10: Feedback
    # ------------------------------------------------------------------
    feedback_path = Path(
        cfg.get("FEEDBACK_PATH", Path.home() / ".playlistgen" / "feedback.json")
    )
    for label, _ in playlists:
        update_feedback(str(feedback_path), label, "generated")

    logging.info(
        "=== Pipeline complete. %d playlists written to %s ===",
        len(playlists),
        cfg.get("OUTPUT_DIR", "./mixes"),
    )
    return playlists


# ---------------------------------------------------------------------------
# Convenience re-exports used by cli.py
# ---------------------------------------------------------------------------


def ensure_tag_mood_cache(cfg: dict, itunes_json: Path) -> Path:
    """Backward-compat shim for cli.py recache-moods command."""
    ensure_tag_cache(cfg, itunes_json)
    return Path(
        cfg.get(
            "TAG_MOOD_CACHE",
            Path.home() / ".playlistgen" / "lastfm_tags_cache.json",
        )
    )
