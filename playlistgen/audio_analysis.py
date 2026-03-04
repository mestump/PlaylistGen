"""
Local audio feature extraction using libROSA.

Extracts BPM (tempo), Energy (RMS mean), SpectralBrightness (spectral centroid mean),
and ZCR (zero-crossing rate mean) from audio files without any external API calls.

Results are cached in a SQLite database keyed by (path, mtime) to avoid
re-analyzing unchanged files. Falls back gracefully if librosa is not installed.

Usage in pipeline.py:
    from .audio_analysis import analyze_library
    df = analyze_library(df, db_path=cfg.get("AUDIO_CACHE_DB", "~/.playlistgen/audio.sqlite"))
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

import pandas as pd
from tqdm import tqdm

try:
    import librosa
    import numpy as np

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logging.warning(
        "librosa not installed — local audio feature extraction (BPM, energy, "
        "spectral brightness) is disabled. Install with: pip install librosa"
    )


_SCHEMA = """
CREATE TABLE IF NOT EXISTS audio_features (
    path      TEXT PRIMARY KEY,
    mtime     REAL,
    bpm       REAL,
    energy    REAL,
    spectral_brightness REAL,
    zcr       REAL,
    analyzed_at INTEGER
);
"""


def _init_db(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(_SCHEMA)
    conn.commit()
    return conn


def _resolve_path(raw: str) -> str:
    """Decode file://localhost URLs to plain filesystem paths."""
    if raw.startswith("file://localhost"):
        return unquote(raw[len("file://localhost") :])
    if raw.startswith("file://"):
        return unquote(raw[len("file://") :])
    return raw


def _cache_get(conn: sqlite3.Connection, path: str, mtime: float) -> Optional[dict]:
    row = conn.execute(
        "SELECT bpm, energy, spectral_brightness, zcr "
        "FROM audio_features WHERE path=? AND mtime=?",
        (path, mtime),
    ).fetchone()
    if row:
        return {
            "bpm": row[0],
            "energy": row[1],
            "spectral_brightness": row[2],
            "zcr": row[3],
        }
    return None


def _cache_set(
    conn: sqlite3.Connection, path: str, mtime: float, features: dict
) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO audio_features
           (path, mtime, bpm, energy, spectral_brightness, zcr, analyzed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            path,
            mtime,
            features.get("bpm"),
            features.get("energy"),
            features.get("spectral_brightness"),
            features.get("zcr"),
            int(time.time()),
        ),
    )
    conn.commit()


def _cache_get_batch(
    conn: sqlite3.Connection, paths_mtimes: list[tuple[str, float]]
) -> dict[str, dict]:
    """Batch cache lookup. Returns {path: {bpm, energy, spectral_brightness, zcr}}."""
    if not paths_mtimes:
        return {}
    result = {}
    # SQLite max variables is 999; batch in chunks of 400 (2 params each)
    chunk_size = 400
    for i in range(0, len(paths_mtimes), chunk_size):
        chunk = paths_mtimes[i : i + chunk_size]
        placeholders = " OR ".join(["(path=? AND mtime=?)"] * len(chunk))
        params = []
        for p, m in chunk:
            params.extend([p, m])
        rows = conn.execute(
            f"SELECT path, bpm, energy, spectral_brightness, zcr "
            f"FROM audio_features WHERE {placeholders}",
            params,
        ).fetchall()
        for row in rows:
            result[row[0]] = {
                "bpm": row[1],
                "energy": row[2],
                "spectral_brightness": row[3],
                "zcr": row[4],
            }
    return result


def _cache_set_batch(
    conn: sqlite3.Connection, records: list[tuple[str, float, dict]]
) -> None:
    """Batch cache write in a single transaction."""
    if not records:
        return
    now = int(time.time())
    conn.executemany(
        """INSERT OR REPLACE INTO audio_features
           (path, mtime, bpm, energy, spectral_brightness, zcr, analyzed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                path,
                mtime,
                features.get("bpm"),
                features.get("energy"),
                features.get("spectral_brightness"),
                features.get("zcr"),
                now,
            )
            for path, mtime, features in records
        ],
    )
    conn.commit()


def analyze_track(file_path: str, duration: int = 120) -> dict:
    """
    Extract audio features from a single audio file using libROSA.

    Analyzes the first `duration` seconds of the track to keep runtime reasonable.

    Returns:
        dict with keys: bpm, energy, spectral_brightness, zcr.
        Returns {} if librosa is not installed or the file cannot be read.
        Never raises.
    """
    if not LIBROSA_AVAILABLE:
        return {}
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True, duration=duration)
        if len(y) == 0:
            return {}

        # BPM via beat tracking
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo) if tempo else None

        # Energy (RMS amplitude mean)
        rms = librosa.feature.rms(y=y)
        energy = float(np.mean(rms))

        # Spectral brightness — spectral centroid normalised by Nyquist frequency
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_brightness = float(np.mean(centroid)) / (sr / 2)

        # Zero-crossing rate (indicator of noisiness / consonance)
        zcr_feat = librosa.feature.zero_crossing_rate(y=y)
        zcr = float(np.mean(zcr_feat))

        return {
            "bpm": bpm,
            "energy": energy,
            "spectral_brightness": spectral_brightness,
            "zcr": zcr,
        }
    except Exception as exc:
        logging.warning("Audio analysis failed for %s: %s", file_path, exc)
        return {}


def _analyze_one(args: tuple) -> tuple:
    """Worker for ThreadPoolExecutor. Returns (idx, path, features)."""
    idx, path, duration = args
    return idx, path, analyze_track(path, duration=duration)


def analyze_library(
    df: pd.DataFrame,
    db_path: str,
    enabled: bool = True,
    workers: int = 0,
    duration: int = 120,
) -> pd.DataFrame:
    """
    Add Energy, SpectralBrightness, ZCR columns to the library DataFrame.

    BPM is already populated from mutagen in itunes.py / metadata.py;
    this adds the three acoustic feature columns that mutagen cannot provide.
    Results are cached in SQLite — only uncached / changed files are analyzed.

    Args:
        df:       Library DataFrame with a 'Location' column.
        db_path:  Path to the SQLite audio cache.
        enabled:  If False, returns df unchanged (LIBROSA_ENABLED=false in config).
        workers:  Parallel analysis threads (default os.cpu_count() or 4).
        duration: Seconds of audio to analyze per track (default 120).

    Returns:
        DataFrame with Energy, SpectralBrightness, ZCR columns added/filled.
    """
    if not enabled:
        return df

    if workers <= 0:
        workers = os.cpu_count() or 4

    db_path = str(Path(db_path).expanduser())
    try:
        conn = _init_db(db_path)
    except Exception as exc:
        logging.warning(
            "Audio cache DB init failed (%s) — skipping audio analysis.", exc
        )
        return df

    # Try to load librosa; analyze_track will return {} if not available
    try:
        df = df.copy()

        # Ensure feature columns exist
        for col in ("Energy", "SpectralBrightness", "ZCR"):
            if col not in df.columns:
                df[col] = None

        # Build work list — resolve paths and get mtimes first
        candidates: list[tuple[int, str, float]] = []  # (idx, path, mtime)

        for idx, row in df.iterrows():
            raw_loc = row.get("Location") or ""
            if not raw_loc:
                continue
            path = _resolve_path(str(raw_loc))
            if not os.path.isfile(path):
                continue
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            candidates.append((idx, path, mtime))

        # Batch cache lookup — single SQL query instead of N individual queries
        cache_map = _cache_get_batch(
            conn, [(path, mtime) for _, path, mtime in candidates]
        )

        to_analyze: list = []
        mtime_map: dict = {}

        for idx, path, mtime in candidates:
            cached = cache_map.get(path)
            if cached is not None:
                for col in ("Energy", "SpectralBrightness", "ZCR"):
                    v = cached.get(col.lower() if col != "ZCR" else "zcr")
                    # Map cache keys to DataFrame columns
                    cache_key = {
                        "Energy": "energy",
                        "SpectralBrightness": "spectral_brightness",
                        "ZCR": "zcr",
                    }[col]
                    v = cached.get(cache_key)
                    if v is not None and df.at[idx, col] is None:
                        df.at[idx, col] = v
            else:
                to_analyze.append((idx, path, duration))
                mtime_map[idx] = mtime

        if not to_analyze:
            logging.info("Audio analysis: all %d tracks loaded from cache.", len(df))
            return df

        logging.info(
            "Audio analysis: analyzing %d new tracks (cached: %d)…",
            len(to_analyze),
            len(df) - len(to_analyze),
        )

        completed = 0
        failed = 0
        cache_batch: list[tuple[str, float, dict]] = []

        # Use ProcessPoolExecutor for CPU-bound librosa work (bypasses GIL)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_analyze_one, t): t for t in to_analyze}
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Analyzing audio",
                unit="track",
                disable=len(to_analyze) < 10,
            ):
                try:
                    idx, path, features = future.result()
                    if features:
                        cache_batch.append((path, mtime_map[idx], features))
                        for col, value in features.items():
                            df.at[idx, col] = value
                    completed += 1
                except Exception as exc:
                    logging.warning("Audio analysis worker error: %s", exc)
                    failed += 1

        # Batch write all results to cache in one transaction
        _cache_set_batch(conn, cache_batch)

    finally:
        conn.close()

    energy_count = df["Energy"].notna().sum() if "Energy" in df.columns else 0
    succeeded = completed - failed
    if LIBROSA_AVAILABLE and not to_analyze:
        logging.info(
            "Audio analysis complete: %d tracks analysed, Energy populated for "
            "%d / %d tracks total.",
            succeeded,
            energy_count,
            len(df),
        )
    elif not LIBROSA_AVAILABLE and failed == 0:
        # All uncached files failed because librosa unavailable — track counts still correct
        logging.warning(
            "Audio analysis complete: %d tracks analysed (librosa unavailable)."
            " Energy populated for %d / %d tracks. Install librosa to enable.",
            completed,
            energy_count,
            len(df),
        )
    elif LIBROSA_AVAILABLE and failed > 0:
        logging.warning(
            "Audio analysis: %d/%d tracks succeeded, %d failed (unreadable format "
            "or codec missing — those tracks will use mood/genre clustering instead).",
            completed - failed,
            len(to_analyze),
            failed,
        )
    elif LIBROSA_AVAILABLE and completed == 0:
        # to_analyze was empty
        logging.info("Audio analysis: no new files to process.")
    else:
        logging.warning(
            "Audio analysis skipped (librosa unavailable, %d uncached tracks).",
            len(to_analyze),
        )
    return df
