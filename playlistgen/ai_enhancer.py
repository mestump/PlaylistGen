"""
Optional Claude AI enhancement for PlaylistGen.

When AI_ENHANCE=true and ANTHROPIC_API_KEY is set, this module:
  1. Summarises each playlist cluster (mood, genre, era, top artists).
  2. Calls the Claude API to generate a creative, evocative playlist name.
  3. Returns a cohesion score (1–10) — clusters scoring below 5 are flagged.

Falls back silently to the original generated label on any error so the
pipeline always completes even without an API key or internet access.

Usage in pipeline.py:
    from .ai_enhancer import enhance_playlists
    labelled = enhance_playlists(labelled, api_key=key, model=model_id)
"""

import json
import logging
from typing import List, Optional, Tuple

import pandas as pd


def _summarise_cluster(df: pd.DataFrame) -> str:
    """
    Build a concise, token-efficient summary of a playlist cluster for the
    Claude prompt.  Only metadata is sent — no file paths or personal data.
    """
    n = len(df)

    def top(col, k=5):
        if col in df.columns and df[col].notna().any():
            counts = df[col].value_counts()
            pairs = [f"{v} ({c})" for v, c in counts.head(k).items()]
            return ", ".join(pairs)
        return "unknown"

    # Era (decade distribution)
    era = "unknown"
    if "Year" in df.columns and df["Year"].notna().any():
        years = pd.to_numeric(df["Year"], errors="coerce").dropna()
        if not years.empty:
            lo, hi = int(years.min()), int(years.max())
            era = f"{lo}–{hi}"

    parts = [
        f"Tracks: {n}",
        f"Mood: {top('Mood', 3)}",
        f"Genre: {top('Genre', 3)}",
        f"Era: {era}",
        f"Top artists: {top('Artist', 5)}",
    ]
    return " | ".join(parts)


def _call_claude(
    summary: str,
    api_key: str,
    model: str,
    client,
) -> Tuple[str, int]:
    """
    Call the Claude API with a playlist summary and return (name, cohesion_score).
    Returns ("", 0) on any error so the caller can fall back gracefully.
    """
    system = (
        "You are an expert music playlist curator. "
        "Given a summary of tracks in a playlist cluster, produce:\n"
        "1. A creative, evocative playlist name (max 5 words, no quotes).\n"
        "2. A cohesion score from 1 (incoherent mix) to 10 (perfectly themed).\n\n"
        "Reply with ONLY a JSON object: {\"name\": \"...\", \"cohesion\": N}"
    )
    user = summary

    try:
        message = client.messages.create(
            model=model,
            max_tokens=64,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        raw = message.content[0].text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        data = json.loads(raw)
        name = str(data.get("name", "")).strip()
        cohesion = int(data.get("cohesion", 5))
        return name, cohesion
    except Exception as exc:
        logging.debug("Claude API call failed: %s", exc)
        return "", 0


def enhance_playlists(
    labelled: List[Tuple[str, pd.DataFrame]],
    api_key: str,
    model: str = "claude-haiku-4-5-20251001",
    min_cohesion: int = 4,
) -> List[Tuple[str, pd.DataFrame]]:
    """
    Enhance playlist labels using the Claude API.

    For each (label, DataFrame) pair:
      - Summarises the cluster metadata.
      - Asks Claude for a creative playlist name and cohesion score.
      - Replaces the label if a valid name is returned.
      - Logs a warning for clusters with low cohesion (< min_cohesion).

    Args:
        labelled:     List of (label, DataFrame) tuples from clustering.
        api_key:      Anthropic API key.
        model:        Claude model ID (default: haiku for speed/cost).
        min_cohesion: Cohesion threshold below which a warning is logged.

    Returns:
        Updated list of (label, DataFrame) tuples with AI-generated names
        where available; original labels where API calls fail.
    """
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except ImportError:
        logging.warning(
            "anthropic package not installed — AI enhancement disabled. "
            "Run: pip install anthropic"
        )
        return labelled
    except Exception as exc:
        logging.warning("Could not initialise Anthropic client: %s", exc)
        return labelled

    enhanced = []
    for label, df in labelled:
        summary = _summarise_cluster(df)
        logging.debug("AI enhancing '%s': %s", label, summary)

        name, cohesion = _call_claude(summary, api_key, model, client)

        if name:
            if cohesion < min_cohesion:
                logging.info(
                    "Low-cohesion cluster (score %d/10): '%s' → '%s'. "
                    "Consider adjusting CLUSTER_COUNT in config.",
                    cohesion, label, name,
                )
            else:
                logging.info(
                    "AI renamed '%s' → '%s' (cohesion %d/10)", label, name, cohesion
                )
            enhanced.append((name, df))
        else:
            enhanced.append((label, df))

    return enhanced


def discover_similar(
    genre: str,
    library_df: pd.DataFrame,
    cfg: dict,
    limit: int = 5,
) -> Optional[Tuple[str, pd.DataFrame]]:
    """
    Discover library tracks similar to popular Spotify playlists for a genre.

    Flow:
      1. Scrape `limit` Spotify playlists matching `genre` via playlist_scraper.
      2. Analyse their track patterns via pattern_analyzer.
      3. Score library tracks by cosine similarity to the scraped patterns.
      4. Return a (label, DataFrame) discovery playlist.

    Returns None if Spotify credentials are not set or no playlists are found.
    """
    client_id = cfg.get("SPOTIFY_CLIENT_ID")
    client_secret = cfg.get("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        logging.warning(
            "SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET not set — "
            "discovery mode requires Spotify API credentials."
        )
        return None

    try:
        from .playlist_scraper import fetch_spotify_playlists
        from .pattern_analyzer import analyze_playlists
        from .similarity import build_vectorizer, score_playlists, playlist_vector

        logging.info("Fetching %d Spotify playlists for '%s'…", limit, genre)
        playlists = fetch_spotify_playlists(genre, limit, client_id, client_secret)
        if not playlists:
            logging.warning("No Spotify playlists found for '%s'.", genre)
            return None

        logging.info("Analysing %d scraped playlists…", len(playlists))
        result = analyze_playlists(playlists, n_clusters=min(3, len(playlists)))
        if not result:
            return None

        vectorizer = build_vectorizer(playlists)
        benchmark_vecs = [playlist_vector(p, vectorizer) for p in playlists]

        # Score every track in the user's library by similarity to scraped playlists
        # We treat each track as a 1-row "playlist" for vectorization
        single_track_dfs = [
            library_df.iloc[[i]] for i in range(len(library_df))
        ]
        scores = score_playlists(single_track_dfs, benchmark_vecs, vectorizer)

        library_df = library_df.copy()
        library_df["_discovery_score"] = scores
        top = (
            library_df.sort_values("_discovery_score", ascending=False)
            .head(50)
            .drop(columns=["_discovery_score"])
            .reset_index(drop=True)
        )

        label = f"Discover: {genre.title()}"
        logging.info("Discovery playlist '%s': %d tracks", label, len(top))
        return label, top

    except Exception as exc:
        logging.warning("Discovery mode failed: %s", exc)
        return None
