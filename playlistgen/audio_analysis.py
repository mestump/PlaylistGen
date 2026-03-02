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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

import pandas as pd

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
        return unquote(raw[len("file://localhost"):])
    if raw.startswith("file://"):
        return unquote(raw[len("file://"):])
    return raw


def _cache_get(
    conn: sqlite3.Connection, path: str, mtime: float
) -> Optional[dict]:
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


def analyze_track(file_path: str) -> dict:
    """
    Extract audio features from a single audio file using libROSA.

    Analyzes the first 120 seconds of the track to keep runtime reasonable.

    Returns:
        dict with keys: bpm, energy, spectral_brightness, zcr.
        Returns {} if librosa is not installed or the file cannot be read.
        Never raises.
    """
    if not LIBROSA_AVAILABLE:
        return {}
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True, duration=120)
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
    idx, path = args
    return idx, path, analyze_track(path)


def analyze_library(
    df: pd.DataFrame,
    db_path: str,
    enabled: bool = True,
    workers: int = 4,
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
        workers:  Parallel analysis threads (default 4).

    Returns:
        DataFrame with Energy, SpectralBrightness, ZCR columns added/filled.
    """
    if not enabled:
        return df

    if not LIBROSA_AVAILABLE:
        logging.warning(
            "librosa is not installed — audio feature extraction skipped. "
            "Clustering will fall back to mood/genre. Install: pip install librosa"
        )
        return df

    db_path = str(Path(db_path).expanduser())
    try:
        conn = _init_db(db_path)
    except Exception as exc:
        logging.warning("Audio cache DB init failed (%s) — skipping audio analysis.", exc)
        return df

    try:
        df = df.copy()

        # Ensure feature columns exist
        for col in ("Energy", "SpectralBrightness", "ZCR"):
            if col not in df.columns:
                df[col] = None

        # Build work list
        to_analyze: list = []
        mtime_map: dict = {}
        path_map: dict = {}

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
            cached = _cache_get(conn, path, mtime)
            if cached:
                if cached.get("energy") is not None:
                    df.at[idx, "Energy"] = cached["energy"]
                if cached.get("spectral_brightness") is not None:
                    df.at[idx, "SpectralBrightness"] = cached["spectral_brightness"]
                if cached.get("zcr") is not None:
                    df.at[idx, "ZCR"] = cached["zcr"]
                # Backfill BPM from audio cache if mutagen missed it
                if cached.get("bpm") and (
                    "BPM" not in df.columns
                    or pd.isnull(df.at[idx, "BPM"])
                    or df.at[idx, "BPM"] == 0
                ):
                    df.at[idx, "BPM"] = cached["bpm"]
            else:
                to_analyze.append((idx, path))
                mtime_map[idx] = mtime
                path_map[idx] = path

        if not to_analyze:
            logging.info(
                "Audio analysis: all %d tracks loaded from cache.", len(df)
            )
            return df

        logging.info(
            "Audio analysis: analyzing %d new tracks (cached: %d)…",
            len(to_analyze),
            len(df) - len(to_analyze),
        )

        completed = 0
        failed = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_analyze_one, t): t for t in to_analyze}
            for future in as_completed(futures):
                try:
                    idx, path, features = future.result()
                except Exception as exc:
                    logging.warning("Audio analysis worker error: %s", exc)
                    failed += 1
                    continue
                if features:
                    if features.get("energy") is not None:
                        df.at[idx, "Energy"] = features["energy"]
                    if features.get("spectral_brightness") is not None:
                        df.at[idx, "SpectralBrightness"] = features["spectral_brightness"]
                    if features.get("zcr") is not None:
                        df.at[idx, "ZCR"] = features["zcr"]
                    if features.get("bpm") and (
                        "BPM" not in df.columns
                        or pd.isnull(df.at[idx, "BPM"])
                        or df.at[idx, "BPM"] == 0
                    ):
                        df.at[idx, "BPM"] = features["bpm"]
                    try:
                        _cache_set(conn, path, mtime_map[idx], features)
                    except Exception as exc:
                        logging.warning("Audio cache write failed for %s: %s", path, exc)
                else:
                    failed += 1
                completed += 1
                if completed % 100 == 0:
                    logging.info(
                        "Audio analysis: %d/%d complete.", completed, len(to_analyze)
                    )

    finally:
        conn.close()
    energy_count = df["Energy"].notna().sum() if "Energy" in df.columns else 0
    succeeded = len(to_analyze) - failed
    if failed:
        logging.warning(
            "Audio analysis: %d/%d tracks succeeded, %d failed (unreadable format "
            "or codec missing — those tracks will use mood/genre clustering instead).",
            succeeded, len(to_analyze), failed,
        )
    else:
        logging.info(
            "Audio analysis complete: %d tracks analysed, Energy populated for "
            "%d / %d tracks total.",
            succeeded, energy_count, len(df),
        )
    return df
