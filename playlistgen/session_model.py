"""
Session-based co-occurrence model from Spotify streaming history JSON.

Replaces Spotify API-based discovery. Works entirely from the user's
personal data export (spotify.com/account/privacy → "Download your data").

Key concepts:
  - Session: a group of plays with no gap > SESSION_GAP_MINUTES between them.
  - Co-occurrence: two tracks in the same session are "similar"; the count
    is used in scoring.py to boost library tracks co-occurring with favorites.
  - Recency: exponential decay means recent plays carry more weight.
    Half-life default is 90 days (a play today = 2× a play 90 days ago).

The model output is used in scoring.py to:
  - Apply a recency multiplier (1.0x → 1.5x) to tracks the user played recently.
  - Add a small co-occurrence bonus for tracks that appeared alongside favorites.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd


def load_streaming_history(
    json_paths: Union[str, List[str], Path],
) -> pd.DataFrame:
    """
    Load one or more Spotify streaming-history JSON files.

    Supports both export formats Spotify has used over the years:
      - Classic:  [{"endTime": "...", "artistName": "...", "trackName": "...", "msPlayed": N}]
      - Extended: [{"ts": "...", "master_metadata_album_artist_name": "...", "ms_played": N}]

    If json_paths is a directory, all *.json files within it are loaded
    automatically (non-streaming files are skipped based on content).

    Returns:
        DataFrame with columns: timestamp (UTC datetime), artist, track,
        ms_played, track_id ("artist - track" lowercased).
    """
    if isinstance(json_paths, (str, Path)):
        p = Path(json_paths)
        if p.is_dir():
            json_paths = sorted(p.glob("*.json"))
        else:
            json_paths = [p]

    records = []
    for path in json_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                continue
            for entry in data:
                if "endTime" in entry:
                    # Classic format
                    records.append(
                        {
                            "timestamp": pd.to_datetime(
                                entry.get("endTime"), utc=True, errors="coerce"
                            ),
                            "artist": entry.get("artistName") or "",
                            "track": entry.get("trackName") or "",
                            "ms_played": int(entry.get("msPlayed") or 0),
                        }
                    )
                elif "ts" in entry:
                    # Extended format
                    records.append(
                        {
                            "timestamp": pd.to_datetime(
                                entry.get("ts"), utc=True, errors="coerce"
                            ),
                            "artist": entry.get(
                                "master_metadata_album_artist_name"
                            )
                            or "",
                            "track": entry.get("master_metadata_track_name")
                            or "",
                            "ms_played": int(entry.get("ms_played") or 0),
                        }
                    )
        except Exception as exc:
            logging.warning(
                "Could not load streaming history from %s: %s", path, exc
            )

    if not records:
        return pd.DataFrame(
            columns=["timestamp", "artist", "track", "ms_played", "track_id"]
        )

    df = pd.DataFrame(records)
    df = df.dropna(subset=["timestamp"])
    df = df[df["artist"].str.strip() != ""]
    df = df[df["track"].str.strip() != ""]
    df["track_id"] = (
        (df["artist"] + " - " + df["track"]).str.lower().str.strip()
    )
    df = df.sort_values("timestamp").reset_index(drop=True)

    logging.info(
        "Streaming history: %d plays, %d unique tracks loaded.",
        len(df),
        df["track_id"].nunique(),
    )
    return df


def build_sessions(
    history_df: pd.DataFrame,
    gap_minutes: int = 30,
    min_ms_played: int = 30_000,
) -> List[List[str]]:
    """
    Group plays into listening sessions.

    A new session starts when the gap between consecutive plays exceeds
    gap_minutes. Plays under min_ms_played (default 30s) are treated as
    skipped tracks and excluded.

    Returns:
        List of sessions, each a list of track_id strings.
    """
    if history_df.empty:
        return []

    df = history_df[history_df["ms_played"] >= min_ms_played].copy()
    if df.empty:
        return []

    gap_ns = gap_minutes * 60 * 1_000_000_000  # nanoseconds

    sessions: List[List[str]] = []
    current: List[str] = []
    prev_ts = None

    for _, row in df.iterrows():
        ts = row["timestamp"]
        track_id = row["track_id"]
        if prev_ts is not None:
            try:
                delta = (ts - prev_ts).value  # nanoseconds
                if delta > gap_ns:
                    if current:
                        sessions.append(current)
                    current = []
            except (AttributeError, TypeError):
                # ts or prev_ts is NaT or an unexpected type — skip gap check
                pass
        current.append(track_id)
        prev_ts = ts

    if current:
        sessions.append(current)

    logging.info(
        "Sessions: %d sessions built from %d plays (gap=%d min).",
        len(sessions),
        len(df),
        gap_minutes,
    )
    return sessions


def build_cooccurrence_matrix(
    sessions: List[List[str]],
) -> Dict[str, Counter]:
    """
    Build a track co-occurrence counter.

    For each unique pair (a, b) appearing in the same session:
        cooccurrence[a][b] += 1
        cooccurrence[b][a] += 1

    Each track is counted at most once per session (set deduplication).

    Returns:
        Dict mapping track_id → Counter of co-occurring track_ids.
    """
    matrix: Dict[str, Counter] = {}
    for session in sessions:
        unique = list(set(session))
        for a in unique:
            for b in unique:
                if a == b:
                    continue
                if a not in matrix:
                    matrix[a] = Counter()
                matrix[a][b] += 1
    return matrix


def recency_scores(
    history_df: pd.DataFrame,
    half_life_days: int = 90,
    now: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute exponential-decay recency scores per track.

    Each play contributes: weight = 2^(-age_days / half_life_days)
    Scores are summed per track then normalised to [0, 1].

    A track played today gets a maximum contribution of 1.0 per play.
    A track played half_life_days ago contributes 0.5 per play.

    Args:
        history_df:     DataFrame with timestamp and track_id columns.
        half_life_days: Decay half-life in days (default 90).
        now:            Reference time as Unix timestamp (defaults to now).

    Returns:
        Dict mapping track_id → normalised recency score [0, 1].
    """
    if history_df.empty:
        return {}

    now_ts = now or time.time()
    scores: Dict[str, float] = {}

    for _, row in history_df.iterrows():
        ts = row.get("timestamp")
        track_id = row.get("track_id")
        if not track_id or pd.isnull(ts):
            continue
        try:
            age_days = (now_ts - ts.timestamp()) / 86400
        except Exception:
            continue
        weight = math.pow(2, -age_days / half_life_days)
        scores[track_id] = scores.get(track_id, 0.0) + weight

    if not scores:
        return {}

    max_score = max(scores.values())
    if max_score > 0:
        scores = {k: v / max_score for k, v in scores.items()}

    return scores


def build_session_model(
    json_paths: Union[str, List[str], Path],
    gap_minutes: int = 30,
    half_life_days: int = 90,
) -> dict:
    """
    High-level: load Spotify JSON files and build the session model.

    Args:
        json_paths:      Path to a Spotify JSON export file or directory of JSON files.
        gap_minutes:     Gap threshold (minutes) for splitting sessions.
        half_life_days:  Recency decay half-life in days.

    Returns:
        dict with keys:
          - 'cooccurrence': Dict[track_id, Counter]
          - 'recency':      Dict[track_id, float] in [0, 1]
          - 'play_counts':  Dict[track_id, int]
        All dicts are empty if loading fails.
    """
    _empty = {"cooccurrence": {}, "recency": {}, "play_counts": {}}
    try:
        history_df = load_streaming_history(json_paths)
        if history_df.empty:
            logging.warning("No streaming history loaded from %s.", json_paths)
            return _empty

        sessions = build_sessions(history_df, gap_minutes=gap_minutes)
        cooccurrence = build_cooccurrence_matrix(sessions)
        recency = recency_scores(history_df, half_life_days=half_life_days)
        play_counts = history_df["track_id"].value_counts().to_dict()

        logging.info(
            "Session model: %d tracks in co-occurrence, %d with recency scores.",
            len(cooccurrence),
            len(recency),
        )
        return {
            "cooccurrence": cooccurrence,
            "recency": recency,
            "play_counts": play_counts,
        }
    except Exception as exc:
        logging.warning("Session model build failed: %s", exc)
        return _empty
