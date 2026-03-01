"""
Audio file metadata extraction using mutagen.

Reads embedded tags (year, BPM, genre, duration, album) from local audio files
so the library DataFrame has accurate data without relying on path parsing.
Supports MP3 (ID3), MP4/M4A, FLAC, OGG, and most other common formats via
mutagen's easy-interface.
"""

import logging
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

import pandas as pd

try:
    from mutagen import File as MutaFile

    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False
    logging.warning(
        "mutagen not installed — audio tag extraction disabled. "
        "Run: pip install mutagen"
    )


def _strip_file_url(path: str) -> str:
    """Decode iTunes-style file://localhost/... URLs to a plain filesystem path."""
    if path.startswith("file://localhost"):
        return unquote(path.replace("file://localhost", ""))
    if path.startswith("file://"):
        return unquote(path.replace("file://", ""))
    return path


def read_audio_tags(file_path: str) -> dict:
    """
    Read embedded audio tags from a file using mutagen's easy interface.

    Returns a dict with keys: year, bpm, genre, duration_sec, album.
    Any field may be None. Never raises — errors are logged at DEBUG level.
    """
    result: dict = {
        "year": None,
        "bpm": None,
        "genre": None,
        "duration_sec": None,
        "album": None,
    }
    if not MUTAGEN_AVAILABLE:
        return result

    resolved = _strip_file_url(file_path)
    if not Path(resolved).exists():
        return result

    try:
        audio = MutaFile(resolved, easy=True)
        if audio is None:
            return result

        # Duration is on audio.info for all formats
        if hasattr(audio, "info") and hasattr(audio.info, "length"):
            result["duration_sec"] = max(0, int(audio.info.length))

        tags = audio.tags
        if tags is None:
            return result

        # Year — try 'date' first (standard EasyID3/EasyMP4/FLAC key), then 'year'
        for key in ("date", "year", "originaldate"):
            val = tags.get(key)
            if val:
                raw = str(val[0]) if isinstance(val, list) else str(val)
                year_str = raw[:4]
                if year_str.isdigit() and 1900 < int(year_str) < 2100:
                    result["year"] = int(year_str)
                    break

        # BPM — 'bpm' is the EasyID3/EasyMP4 key; some files use 'tempo'
        for key in ("bpm", "tempo"):
            val = tags.get(key)
            if val:
                raw = str(val[0]) if isinstance(val, list) else str(val)
                try:
                    bpm = float(raw.replace(",", ".").split(".")[0])
                    if 40 < bpm < 300:
                        result["bpm"] = int(bpm)
                    break
                except (ValueError, TypeError):
                    pass

        # Genre
        val = tags.get("genre")
        if val:
            raw = str(val[0]) if isinstance(val, list) else str(val)
            raw = raw.strip()
            # ID3 genre tags can be numeric codes like "(17)" — skip those
            if raw and not raw.startswith("("):
                result["genre"] = raw

        # Album
        val = tags.get("album")
        if val:
            raw = str(val[0]) if isinstance(val, list) else str(val)
            if raw.strip():
                result["album"] = raw.strip()

    except Exception as exc:
        logging.debug("mutagen tag read failed for %s: %s", file_path, exc)

    return result


def enrich_dataframe(df: pd.DataFrame, enabled: bool = True) -> pd.DataFrame:
    """
    Add/fill Year, BPM, Genre, Duration, Album columns from embedded audio tags.

    Reads tags for every row with a non-null Location. Existing values (e.g.
    from an iTunes XML export) are preserved — mutagen only fills gaps.

    Args:
        df:      Library DataFrame. Must have a 'Location' column.
        enabled: If False (or mutagen unavailable), returns df unchanged.

    Returns:
        DataFrame with Year, BPM, Duration, Album columns added where missing.
    """
    if not enabled or not MUTAGEN_AVAILABLE:
        return df

    df = df.copy()

    # Ensure target columns exist
    for col in ("Year", "BPM", "Duration", "Album"):
        if col not in df.columns:
            df[col] = None

    mask = (
        df["Location"].notna()
        & (df["Location"].astype(str).str.strip() != "")
        & (df["Location"].astype(str).str.lower() != "nan")
    )
    rows_to_enrich = df[mask]
    if rows_to_enrich.empty:
        return df

    logging.info(
        "Enriching %d tracks with embedded audio tags...", len(rows_to_enrich)
    )

    for idx, row in rows_to_enrich.iterrows():
        tags = read_audio_tags(str(row["Location"]))

        # Only fill genuinely missing values (NaN / None / empty string)
        def _is_missing(val) -> bool:
            if val is None:
                return True
            try:
                import math
                return math.isnan(float(val))
            except (TypeError, ValueError):
                return str(val).strip() in ("", "None", "nan")

        if tags["year"] and _is_missing(df.at[idx, "Year"]):
            df.at[idx, "Year"] = tags["year"]
        if tags["bpm"] and _is_missing(df.at[idx, "BPM"]):
            df.at[idx, "BPM"] = tags["bpm"]
        if tags["genre"] and _is_missing(df.at[idx, "Genre"]):
            df.at[idx, "Genre"] = tags["genre"]
        if tags["duration_sec"] and _is_missing(df.at[idx, "Duration"]):
            df.at[idx, "Duration"] = tags["duration_sec"]
        if tags["album"] and _is_missing(df.at[idx, "Album"]):
            df.at[idx, "Album"] = tags["album"]

    # Coerce numeric columns
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["BPM"] = pd.to_numeric(df["BPM"], errors="coerce")
    df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce")

    valid_years = df["Year"].notna().sum()
    valid_bpm = df["BPM"].notna().sum()
    logging.info(
        "Audio tag enrichment complete. Year: %d tracks, BPM: %d tracks.",
        valid_years,
        valid_bpm,
    )
    return df
