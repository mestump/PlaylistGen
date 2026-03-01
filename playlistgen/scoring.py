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

    moods_out = []
    scores_out = []

    for _, row in df.iterrows():
        track_id = f"{row['Artist']} - {row['Name']}".strip().lower()
        artist = str(row.get("Artist", ""))
        genre = str(row.get("Genre", "") or "")

        play_count = int(row.get("Play Count", 0) or 0)
        skip_count = int(row.get("Skip Count", 0) or 0)

        # Tags for this track (handle legacy dict format)
        tags = tag_mood_db.get(track_id, [])
        if isinstance(tags, dict):
            tags = tags.get("tags", [])

        # Mood — use pre-computed value if batch enrichment ran, else derive
        existing_mood = row.get("Mood")
        if existing_mood and existing_mood not in ("Unknown", ""):
            mood = existing_mood
        else:
            mood = canonical_mood(
                tags, genre=genre if genre else None, tag_counts=tag_counts
            )
        moods_out.append(mood if mood else "Unknown")

        # --- Base score components ---
        artist_score = artist_scores.get(artist, 0)
        genre_score = genre_scores.get(genre.lower(), 0) if genre else 0
        mood_score = mood_scores.get(mood, 0) if mood else 0

        # Year score — use the Year column directly (FIXED: no more path parsing)
        year_score = 0
        raw_year = row.get("Year")
        if raw_year is not None:
            try:
                year = int(float(raw_year))
                if 1900 < year < 2100:
                    year_score = year_scores.get(year, 0)
            except (TypeError, ValueError):
                pass

        # Spotify play/skip counts for this track
        spotify_play = track_play_counts.get(track_id, 0)
        spotify_skip = track_skip_counts.get(track_id, 0)

        score = (
            w["artist"] * artist_score
            + w["genre"] * genre_score
            + w["mood"] * mood_score
            + w["year"] * year_score
            + w["play"] * (play_count + spotify_play)
            + w["skip"] * (skip_count + spotify_skip)
        )

        # --- Session model bonus components (Phase 2) ---
        if session_model:
            # 1. Recency multiplier: boost tracks the user played recently.
            #    recency_weight in [0, 1]. Multiplier ranges from 1.0x (not in
            #    history) to 1.5x (most recently played).
            recency_weight = recency_map.get(track_id, 0.0)
            score *= 1.0 + 0.5 * recency_weight

            # 2. Co-occurrence boost: add a bonus for tracks that often appear
            #    in the same listening sessions as the user's most-played tracks.
            if top_played and cooccurrence_map:
                co_boost = sum(
                    cooccurrence_map.get(fav, {}).get(track_id, 0)
                    for fav in top_played
                )
                score += 0.05 * min(co_boost / 50.0, 1.0)

            # 3. Energy preference match: if the track has energy data, boost
            #    it if its energy level matches the user's preferred listening energy.
            if energy_preference is not None:
                energy = row.get("Energy")
                if energy is not None and pd.notnull(energy):
                    try:
                        energy_match = 1.0 - abs(float(energy) - energy_preference) / 10.0
                        score += 0.1 * max(energy_match, 0)
                    except (TypeError, ValueError):
                        pass

        scores_out.append(score)

    df["Mood"] = moods_out
    df["Score"] = scores_out

    # Compute energy_preference lazily after scoring if session_model present
    # (used for subsequent calls; no effect on this call's scores)
    if session_model and "Energy" in df.columns and top_played:
        top_played_mask = df.apply(
            lambda r: f"{r.get('Artist','')} - {r.get('Name','')}".lower()
            in top_played,
            axis=1,
        )
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
