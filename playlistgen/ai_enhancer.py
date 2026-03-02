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
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from .utils import progress_bar


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


# ---------------------------------------------------------------------------
# Claude batch metadata enrichment
# ---------------------------------------------------------------------------

_ENRICH_SCHEMA = """
CREATE TABLE IF NOT EXISTS claude_enrichment (
    key         TEXT PRIMARY KEY,
    mood        TEXT,
    energy      INTEGER,
    valence     INTEGER,
    tags        TEXT,
    enriched_at INTEGER
);
"""

_ENRICH_SYSTEM = (
    "You are a music metadata expert. Given a numbered list of tracks, classify "
    "each one's mood, energy, and valence.\n\n"
    "Respond with ONLY a JSON array of objects, one per track:\n"
    "[{\"idx\": N, \"mood\": \"Mood\", \"energy\": E, \"valence\": V, "
    "\"tags\": [\"tag1\", \"tag2\"]}]\n\n"
    "Mood must be exactly one of: Happy, Sad, Angry, Chill, Energetic, "
    "Romantic, Epic, Dreamy, Groovy, Nostalgic\n"
    "Energy: integer 1-10 (1=very quiet/calm, 10=very loud/intense)\n"
    "Valence: integer 1-10 (1=very dark/negative, 10=very bright/positive)\n"
    "Tags: list of 3-5 short descriptive music tags"
)


def _init_enrich_db(db_path: str):
    import sqlite3

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(_ENRICH_SCHEMA)
    conn.commit()
    return conn


def batch_enrich_metadata(
    df: pd.DataFrame,
    api_key: str,
    model: str = "claude-haiku-4-5-20251001",
    batch_size: int = 150,
    cache_db: str = None,
    rate_limit_ms: int = 0,
) -> pd.DataFrame:
    """
    Batch-enrich track metadata (Mood, Energy, Valence) using Claude.

    Sends tracks in batches to Claude Haiku for mood/energy classification.
    Results are cached in SQLite; subsequent runs skip already-enriched tracks.

    Cost estimate: ~$0.001/track with Haiku (one-time; cached thereafter).

    Args:
        df:             Library DataFrame with Artist, Name, Genre, BPM columns.
        api_key:        Anthropic API key.
        model:          Claude model ID (Haiku recommended for cost/speed).
        batch_size:     Tracks per API call (default 150).
        cache_db:       SQLite cache path.
        rate_limit_ms:  Minimum ms between API calls (0 = no pacing). Set e.g.
                        1000 to send at most one batch per second and avoid
                        exhausting Claude's RPM quota.

    Returns:
        DataFrame with Mood, Energy, Valence columns filled where previously missing.
    """
    if cache_db is None:
        cache_db = str(
            Path.home() / ".playlistgen" / "claude_enrichment.sqlite"
        )
    cache_db = str(Path(cache_db).expanduser())

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
    except ImportError:
        logging.warning(
            "anthropic package not installed — batch enrichment skipped."
        )
        return df
    except Exception as exc:
        logging.warning("Anthropic client init failed: %s", exc)
        return df

    try:
        conn = _init_enrich_db(cache_db)
    except Exception as exc:
        logging.warning(
            "Enrichment cache DB init failed (%s) — proceeding without cache.", exc
        )
        conn = None

    df = df.copy()
    for col in ("Mood", "Energy", "Valence"):
        if col not in df.columns:
            df[col] = None

    # Build list of tracks that still need enrichment
    to_enrich = []  # (df_idx, cache_key, label_string)
    for idx, row in df.iterrows():
        artist = str(row.get("Artist") or "")
        name = str(row.get("Name") or "")
        if not artist or not name:
            continue
        key = f"{artist} - {name}".lower().strip()

        # Skip if Mood already populated from Last.fm / mood_map
        if pd.notnull(row.get("Mood")) and row.get("Mood") not in (
            "Unknown",
            "",
            None,
        ):
            continue

        # Check SQLite cache
        if conn:
            cached = conn.execute(
                "SELECT mood, energy, valence FROM claude_enrichment WHERE key=?",
                (key,),
            ).fetchone()
            if cached:
                if cached[0]:
                    df.at[idx, "Mood"] = cached[0]
                if cached[1] is not None:
                    df.at[idx, "Energy"] = int(cached[1])
                if cached[2] is not None:
                    df.at[idx, "Valence"] = int(cached[2])
                continue

        genre = str(row.get("Genre") or "")
        bpm = row.get("BPM")
        bpm_str = (
            f" | BPM:{int(bpm)}"
            if bpm is not None and pd.notnull(bpm) and bpm > 0
            else ""
        )
        genre_str = f" | Genre:{genre}" if genre else ""
        label = f"{artist} - {name}{genre_str}{bpm_str}"
        to_enrich.append((idx, key, label))

    if not to_enrich:
        logging.info("Batch enrichment: all tracks already enriched (cached).")
        if conn:
            conn.close()
        return df

    enriched_count = 0
    num_batches = (len(to_enrich) + batch_size - 1) // batch_size
    logging.info(
        "Batch enrichment: %d tracks in %d batches of up to %d…",
        len(to_enrich), num_batches, batch_size,
    )

    _last_batch_time: float = 0.0
    bar = progress_bar(range(num_batches), desc="Claude enrichment", total=num_batches)

    for batch_num in bar:
        batch_start = batch_num * batch_size
        batch = to_enrich[batch_start : batch_start + batch_size]

        # Inter-batch rate limiting
        if rate_limit_ms > 0:
            elapsed = time.time() - _last_batch_time
            wait = (rate_limit_ms / 1000.0) - elapsed
            if wait > 0:
                time.sleep(wait)

        user_msg = "Classify these tracks:\n" + "\n".join(
            f"{i + 1}. {item[2]}" for i, item in enumerate(batch)
        )

        # Retry with exponential backoff: 2s, 4s, 8s
        results = None
        for attempt in range(3):
            try:
                message = client.messages.create(
                    model=model,
                    max_tokens=8192,
                    system=_ENRICH_SYSTEM,
                    messages=[{"role": "user", "content": user_msg}],
                )
                _last_batch_time = time.time()
                raw = message.content[0].text.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1].lstrip("json").strip()
                results = json.loads(raw)
                if isinstance(results, dict):
                    results = results.get("results", results.get("tracks", []))
                break
            except Exception as exc:
                backoff = 2 ** (attempt + 1)  # 2, 4, 8 seconds
                if attempt < 2:
                    logging.warning(
                        "Claude enrichment batch %d/%d failed (attempt %d/3): %s. "
                        "Retrying in %ds…",
                        batch_num + 1, num_batches, attempt + 1, exc, backoff,
                    )
                    time.sleep(backoff)
                else:
                    logging.warning(
                        "Claude enrichment batch %d/%d failed after 3 attempts: %s. "
                        "Skipping batch.",
                        batch_num + 1, num_batches, exc,
                    )

        if results is None:
            continue

        for item in results:
            if not isinstance(item, dict):
                continue
            try:
                idx_1based = int(item.get("idx", 0))
            except (TypeError, ValueError):
                continue
            if not (1 <= idx_1based <= len(batch)):
                continue
            orig_idx, key, _ = batch[idx_1based - 1]
            mood = str(item.get("mood") or "").strip()
            energy = item.get("energy")
            valence = item.get("valence")
            tags = item.get("tags", [])

            if mood:
                df.at[orig_idx, "Mood"] = mood
            if energy is not None:
                try:
                    df.at[orig_idx, "Energy"] = int(energy)
                except (ValueError, TypeError):
                    pass
            if valence is not None:
                try:
                    df.at[orig_idx, "Valence"] = int(valence)
                except (ValueError, TypeError):
                    pass

            if conn:
                try:
                    conn.execute(
                        """INSERT OR REPLACE INTO claude_enrichment
                           (key, mood, energy, valence, tags, enriched_at)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            key,
                            mood or None,
                            int(energy) if energy is not None else None,
                            int(valence) if valence is not None else None,
                            json.dumps(tags) if tags else None,
                            int(time.time()),
                        ),
                    )
                except Exception:
                    pass
            enriched_count += 1

        # Commit once per batch (not per result)
        if conn:
            try:
                conn.commit()
            except Exception:
                pass

    if conn:
        conn.close()

    mood_coverage = (
        df["Mood"].notna() & (df["Mood"] != "Unknown")
    ).sum()
    logging.info(
        "Batch enrichment: %d tracks enriched; Mood populated for %d / %d.",
        enriched_count,
        mood_coverage,
        len(df),
    )
    return df


# ---------------------------------------------------------------------------
# Claude full playlist curation
# ---------------------------------------------------------------------------

_CURATE_SYSTEM_TMPL = (
    "You are an expert music playlist curator. Given a numbered list of tracks "
    "with metadata, group them into {n} cohesive, themed playlists.\n\n"
    "Rules:\n"
    "- Create exactly {n} playlists\n"
    "- Each track appears in AT MOST one playlist (some may be unassigned)\n"
    "- Each playlist should have 15-50 tracks\n"
    "- Consider: mood, genre, energy level, era, and overall sonic coherence\n"
    "- Playlist names must be creative and evocative (max 5 words)\n\n"
    "Respond with ONLY a JSON object:\n"
    '{{\"playlists\": [{{\"name\": \"playlist name\", '
    '\"track_ids\": [1, 7, 23], \"theme\": \"one-line description\"}}]}}'
)


def claude_curate_playlists(
    scored_df: pd.DataFrame,
    n_playlists: int,
    api_key: str,
    model: str = "claude-sonnet-4-6",
    max_tracks: int = 300,
) -> List[Tuple[str, pd.DataFrame]]:
    """
    Use Claude to curate playlists directly from scored track metadata.

    Sends the top `max_tracks` scored tracks to Claude Sonnet, which groups
    them into n_playlists themed playlists with creative names. This is an
    alternative to algorithmic clustering (enabled via AI_CURATE=true in config).

    Args:
        scored_df:    DataFrame with Score, Mood, Genre, Artist, Name, Year, Energy, BPM.
        n_playlists:  Number of playlists to generate.
        api_key:      Anthropic API key.
        model:        Claude model (Sonnet recommended for curation quality).
        max_tracks:   Maximum tracks to send to Claude (default 300).

    Returns:
        List of (label, DataFrame) tuples. Returns [] on any error.
    """
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
    except ImportError:
        logging.warning("anthropic package not installed — AI curation skipped.")
        return []
    except Exception as exc:
        logging.warning("Anthropic client init failed for curation: %s", exc)
        return []

    top_df = (
        scored_df.sort_values("Score", ascending=False)
        .head(max_tracks)
        .reset_index(drop=True)
    )
    if top_df.empty:
        return []

    # Build numbered track list
    lines = []
    for i, (_, row) in enumerate(top_df.iterrows(), start=1):
        artist = str(row.get("Artist") or "")
        name = str(row.get("Name") or "")
        parts = [f"{artist} - {name}"]
        mood = str(row.get("Mood") or "")
        if mood and mood not in ("Unknown", ""):
            parts.append(f"Mood:{mood}")
        genre = str(row.get("Genre") or "")
        if genre:
            parts.append(f"Genre:{genre}")
        energy = row.get("Energy")
        if energy is not None and pd.notnull(energy):
            parts.append(f"Energy:{energy:.1f}")
        bpm = row.get("BPM")
        if bpm is not None and pd.notnull(bpm) and bpm > 0:
            parts.append(f"BPM:{int(bpm)}")
        year = row.get("Year")
        if year is not None and pd.notnull(year):
            try:
                parts.append(f"Era:{int(year)}")
            except (ValueError, TypeError):
                pass
        lines.append(f"{i}. {' | '.join(parts)}")

    user_msg = (
        f"Group these {len(top_df)} tracks into {n_playlists} playlists:\n\n"
        + "\n".join(lines)
    )
    system_msg = _CURATE_SYSTEM_TMPL.format(n=n_playlists)

    logging.info(
        "Claude curation: sending %d tracks, requesting %d playlists…",
        len(top_df),
        n_playlists,
    )

    try:
        message = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_msg,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        data = json.loads(raw)
        playlists_data = data.get("playlists", [])
    except Exception as exc:
        logging.warning("Claude curation API call failed: %s", exc)
        return []

    result: List[Tuple[str, pd.DataFrame]] = []
    assigned: set = set()

    for pl in playlists_data:
        if not isinstance(pl, dict):
            continue
        name = str(pl.get("name") or "").strip() or "Curated Mix"
        track_ids = pl.get("track_ids", [])
        if not track_ids:
            continue

        indices = []
        for tid in track_ids:
            try:
                idx = int(tid) - 1  # 1-based → 0-based
                if 0 <= idx < len(top_df) and idx not in assigned:
                    indices.append(idx)
                    assigned.add(idx)
            except (ValueError, TypeError):
                continue

        if not indices:
            continue

        playlist_df = top_df.iloc[indices].reset_index(drop=True)
        result.append((name, playlist_df))
        logging.info(
            "Claude curated '%s': %d tracks", name, len(playlist_df)
        )

    logging.info(
        "Claude curation complete: %d playlists, %d / %d tracks assigned.",
        len(result),
        len(assigned),
        len(top_df),
    )
    return result
