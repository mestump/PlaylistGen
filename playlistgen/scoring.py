"""
Track scoring for PlaylistGen.

Assigns a numerical Score to each track based on:
  - Artist affinity (from Spotify listening history)
  - Genre affinity (FIXED: uses genre_scores, which is now populated correctly)
  - Mood affinity (from canonical_mood via mood_map — 500+ keywords)
  - Year affinity (FIXED: reads the Year column, not file-path parsing)
  - Play count (iTunes + Spotify combined)
  - Skip count (iTunes + Spotify combined — penalises skipped tracks)
  - Recency multiplier (Phase 2: from session_model recency scores)
  - Co-occurrence boost (Phase 2: from session_model co-occurrence matrix)
  - Energy preference match (Phase 2: from session_model + audio features)

Also populates a "Mood" column on the DataFrame for use in clustering.
"""

import logging

import pandas as pd

from .config import load_config
from .mood_map import canonical_mood, build_tag_counts

logging.basicConfig(level=logging.INFO)

_DEFAULT_WEIGHTS = {
    "artist": 2.0,
    "genre": 1.0,
    "mood": 1.0,
    "year": 0.5,
    "play": 2.0,
    "skip": -3.0,
}


def score_tracks(
    itunes_df: pd.DataFrame,
    config=None,
    tag_mood_db: dict = None,
    weights: dict = None,
    session_model: dict = None,
) -> pd.DataFrame:
    """
    Add 'Score' and 'Mood' columns to the library DataFrame.

    Args:
        itunes_df:     Library DataFrame (from load_itunes_json or build_library_from_dir).
        config:        Either a taste-profile dict or a config dict with PROFILE_PATH.
                       If None, tries to load the profile from disk.
        tag_mood_db:   Dict mapping "artist - track" → List[str] of Last.fm tags.
                       If None, tries to load from the SQLite/JSON cache.
        weights:       Scoring weight overrides.
        session_model: Optional dict from session_model.build_session_model() with keys:
                       'cooccurrence', 'recency', 'play_counts'. Provides recency
                       multipliers and co-occurrence boosts.

    Returns:
        Copy of itunes_df with 'Score' and 'Mood' columns added.
    """
    # --- Resolve profile ---
    if isinstance(config, dict) and "PROFILE_PATH" in config:
        cfg = config
        from .spotify_profile import load_profile
        profile = load_profile(cfg["PROFILE_PATH"])
    else:
        cfg = load_config()
        if isinstance(config, dict):
            profile = config
        else:
            from .spotify_profile import load_profile
            profile = load_profile() if config is None else {}

    # --- Resolve tag DB ---
    if tag_mood_db is None:
        from .tag_mood_service import load_tag_mood_db
        tag_mood_db = load_tag_mood_db()

    # --- Weights ---
    w = {**_DEFAULT_WEIGHTS, **(weights or {})}

    # --- Pre-compute tag counts for IDF weighting in canonical_mood() ---
    tag_counts = build_tag_counts(tag_mood_db)

    # --- Pre-compute session model components ---
    recency_map: dict = {}
    cooccurrence_map: dict = {}
    top_played: list = []
    energy_preference: float = None  # type: ignore[assignment]

    if session_model:
        recency_map = session_model.get("recency", {})
        cooccurrence_map = session_model.get("cooccurrence", {})
        play_counts = session_model.get("play_counts", {})
        # Top-50 most-played tracks in streaming history
        top_played = sorted(play_counts, key=play_counts.get, reverse=True)[:50]

    # --- Score each track ---
    df = itunes_df.copy()

    artist_scores = profile.get("artist_scores", {})
    genre_scores = profile.get("genre_scores", {})  # now correctly populated
    mood_scores = profile.get("mood_scores", {})
    # year_scores keys are stored as strings (JSON) — normalise to int
    year_scores = {
        int(k): v
        for k, v in profile.get("year_scores", {}).items()
        if str(k).isdigit()
    }
    track_play_counts = profile.get("track_play_counts", {})
    track_skip_counts = profile.get("track_skip_counts", {})

    # --- Compute track IDs vectorized ---
    df["_track_id"] = (df["Artist"].astype(str) + " - " + df["Name"].astype(str)).str.strip().str.lower()

    # --- Compute moods (requires per-row tag lookup, but avoid iterrows) ---
    def _resolve_mood(row):
        existing = row.get("Mood")
        if existing and existing not in ("Unknown", ""):
            return existing
        tid = row["_track_id"]
        tags = tag_mood_db.get(tid, [])
        if isinstance(tags, dict):
            tags = tags.get("tags", [])
        genre = str(row.get("Genre", "") or "")
        mood = canonical_mood(tags, genre=genre if genre else None, tag_counts=tag_counts)
        return mood if mood else "Unknown"

    df["Mood"] = df.apply(_resolve_mood, axis=1)

    # --- Vectorized score computation ---
    df["_artist_score"] = df["Artist"].map(artist_scores).fillna(0)
    df["_genre_score"] = df["Genre"].fillna("").str.lower().map(genre_scores).fillna(0)
    df["_mood_score"] = df["Mood"].map(mood_scores).fillna(0)

    # Year score — vectorized
    df["_year_int"] = pd.to_numeric(df.get("Year"), errors="coerce")
    df["_year_score"] = df["_year_int"].map(year_scores).fillna(0)
    # Zero out invalid years
    df.loc[~df["_year_int"].between(1901, 2099), "_year_score"] = 0

    # Play/skip counts
    play_col = pd.to_numeric(df.get("Play Count"), errors="coerce").fillna(0)
    skip_col = pd.to_numeric(df.get("Skip Count"), errors="coerce").fillna(0)
    spotify_play_col = df["_track_id"].map(track_play_counts).fillna(0)
    spotify_skip_col = df["_track_id"].map(track_skip_counts).fillna(0)

    df["Score"] = (
        w["artist"] * df["_artist_score"]
        + w["genre"] * df["_genre_score"]
        + w["mood"] * df["_mood_score"]
        + w["year"] * df["_year_score"]
        + w["play"] * (play_col + spotify_play_col)
        + w["skip"] * (skip_col + spotify_skip_col)
    )

    # --- Session model bonus (vectorized where possible) ---
    if session_model:
        # 1. Recency multiplier
        recency_col = df["_track_id"].map(recency_map).fillna(0)
        df["Score"] = df["Score"] * (1.0 + 0.5 * recency_col)

        # 2. Co-occurrence boost
        if top_played and cooccurrence_map:
            def _co_boost(tid):
                return sum(
                    cooccurrence_map.get(fav, {}).get(tid, 0)
                    for fav in top_played
                )
            co_col = df["_track_id"].map(_co_boost)
            df["Score"] = df["Score"] + 0.05 * (co_col / 50.0).clip(upper=1.0)

        # 3. Energy preference match
        if energy_preference is not None and "Energy" in df.columns:
            energy_col = pd.to_numeric(df["Energy"], errors="coerce")
            energy_match = (1.0 - (energy_col - energy_preference).abs() / 10.0).clip(lower=0)
            df["Score"] = df["Score"] + 0.1 * energy_match.fillna(0)

    # Clean up temporary columns
    df.drop(columns=["_track_id", "_artist_score", "_genre_score", "_mood_score",
                      "_year_int", "_year_score"], inplace=True)

    # Compute energy_preference lazily after scoring if session_model present
    if session_model and "Energy" in df.columns and top_played:
        track_ids = (df["Artist"].astype(str) + " - " + df["Name"].astype(str)).str.lower()
        top_set = set(top_played)
        top_played_mask = track_ids.isin(top_set)
        energy_vals = pd.to_numeric(
            df.loc[top_played_mask, "Energy"], errors="coerce"
        ).dropna()
        if not energy_vals.empty:
            logging.debug(
                "Energy preference from session history: %.2f", energy_vals.mean()
            )

    # Diagnostics
    scored = (df["Score"] > 0).sum()
    zero = (df["Score"] == 0).sum()
    mood_coverage = (df["Mood"] != "Unknown").sum()
    logging.info(
        "Scoring complete: %d tracks >0, %d zero, %d mood-tagged, of %d total.",
        scored,
        zero,
        mood_coverage,
        len(df),
    )
    if zero > len(df) * 0.5:
        logging.warning(
            "More than 50%% of tracks scored zero. "
            "If you have no Spotify history this is expected — "
            "play counts will still drive ordering."
        )

    return df


def top_tracks(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return the n highest-scoring tracks (for debugging)."""
    return df.sort_values("Score", ascending=False).head(n)
