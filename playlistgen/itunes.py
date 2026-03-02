"""
iTunes library loading and local directory scanning for PlaylistGen.

Provides three entry points:
  convert_itunes_xml()    — Parse iTunes XML plist → slim JSON file.
  load_itunes_json()      — Load that JSON into a DataFrame (adds Year, BPM, Duration, Album).
  build_library_from_dir() — Scan a local directory for audio files using mutagen.
  save_itunes_json()      — Persist a DataFrame back to the slim JSON format.
"""

import json
import logging
import datetime
import plistlib
from pathlib import Path
from urllib.parse import unquote

import pandas as pd

from .utils import sanitize_label  # noqa: F401 — re-exported for backward compat


# ---------------------------------------------------------------------------
# iTunes XML → JSON
# ---------------------------------------------------------------------------


def _decode_location(raw: str) -> str:
    """Convert an iTunes file:// URL to a plain filesystem path."""
    if raw.startswith("file://localhost"):
        return unquote(raw.replace("file://localhost", ""))
    if raw.startswith("file://"):
        return unquote(raw.replace("file://", ""))
    return raw


def _convert_datetimes(obj):
    """Recursively convert datetime objects in dicts/lists to ISO strings."""
    if isinstance(obj, dict):
        return {k: _convert_datetimes(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_datetimes(i) for i in obj]
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    return obj


def convert_itunes_xml(in_path: str, out_path: str) -> None:
    """
    Convert an iTunes Music Library XML (plist) to a slim JSON file.

    Decodes file:// URLs in the Location field to plain paths.
    Converts datetime objects to ISO strings (JSON-serialisable).
    """
    with open(in_path, "rb") as f:
        plist = plistlib.load(f)
    tracks = list(plist.get("Tracks", {}).values())
    tracks = _convert_datetimes(tracks)

    # Decode file:// URLs in place so downstream code gets plain paths
    for t in tracks:
        if "Location" in t and isinstance(t["Location"], str):
            t["Location"] = _decode_location(t["Location"])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"tracks": tracks}, f, ensure_ascii=False, indent=2)
    logging.info("Converted %d iTunes tracks to %s", len(tracks), out_path)


# ---------------------------------------------------------------------------
# Load iTunes JSON → DataFrame
# ---------------------------------------------------------------------------

_ITUNES_COL_MAP = {
    # iTunes XML field name  →  canonical DataFrame column name
    "Name": "Name",
    "Title": "Name",
    "Track Name": "Name",
    "Artist": "Artist",
    "Genre": "Genre",
    "Location": "Location",
    "Play Count": "Play Count",
    "Skip Count": "Skip Count",
    "Year": "Year",
    "BPM": "BPM",
    "Total Time": "Duration",   # Total Time is milliseconds in iTunes XML
    "Album": "Album",
}

_KEEP_COLS = [
    "Name", "Artist", "Genre", "Location",
    "Play Count", "Skip Count", "Year", "BPM", "Duration", "Album",
]


def load_itunes_json(path: str) -> pd.DataFrame:
    """
    Load a slim iTunes JSON file into a normalised pandas DataFrame.

    Preserves Year, BPM, Duration (converted from ms → seconds), and Album
    in addition to the original columns so scoring and clustering work properly.
    Decodes any remaining file:// URLs in the Location column.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    arr = data.get("tracks", data)
    df = pd.DataFrame(arr)

    # Rename columns to canonical names
    rename = {
        old: new
        for old, new in _ITUNES_COL_MAP.items()
        if old in df.columns and old != new
    }
    df = df.rename(columns=rename)

    # Keep only the columns we care about (ignore columns not present)
    cols = [c for c in _KEEP_COLS if c in df.columns]
    df = df[cols]

    # Drop rows missing essential data
    df.dropna(subset=["Name", "Artist"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Clean up Genre
    if "Genre" in df.columns:
        df["Genre"] = (
            df["Genre"].fillna("").astype(str).str.strip().str.title()
        )
        df.loc[df["Genre"] == "", "Genre"] = None

    # Ensure play/skip counts are ints
    for col in ("Play Count", "Skip Count"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Year: coerce to int, drop out-of-range values
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df.loc[~df["Year"].between(1900, 2100, inclusive="both"), "Year"] = None

    # BPM: coerce to float
    if "BPM" in df.columns:
        df["BPM"] = pd.to_numeric(df["BPM"], errors="coerce")
        df.loc[~df["BPM"].between(40, 300), "BPM"] = None

    # Duration: iTunes stores Total Time in milliseconds — convert to seconds
    if "Duration" in df.columns:
        df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce")
        # Heuristic: values > 10000 are almost certainly milliseconds
        big = df["Duration"].notna() & (df["Duration"] > 10000)
        df.loc[big, "Duration"] = (df.loc[big, "Duration"] / 1000).round()

    # Decode any remaining file:// URLs in Location
    if "Location" in df.columns:
        df["Location"] = df["Location"].apply(
            lambda x: _decode_location(str(x)) if pd.notnull(x) else x
        )

    logging.info(
        "Loaded %d tracks from %s  (year: %d, bpm: %d)",
        len(df),
        path,
        df["Year"].notna().sum() if "Year" in df.columns else 0,
        df["BPM"].notna().sum() if "BPM" in df.columns else 0,
    )
    return df


# ---------------------------------------------------------------------------
# Local directory scan
# ---------------------------------------------------------------------------

AUDIO_EXTS = {".mp3", ".m4a", ".flac", ".ogg", ".wav", ".aac", ".wma", ".opus"}


def build_library_from_dir(directory: str, mutagen_enabled: bool = True) -> pd.DataFrame:
    """
    Recursively scan a directory for audio files and build a library DataFrame.

    Tries to read embedded tags (year, BPM, genre, duration, album) via mutagen.
    Falls back to parsing the filename as "Artist - Title" when mutagen returns
    no artist/title, which is common for untagged files.

    Args:
        directory:       Path to the music folder to scan recursively.
        mutagen_enabled: If False, skip embedded tag extraction (respects
                         MUTAGEN_ENABLED config flag).
    """
    from .metadata import enrich_dataframe, MUTAGEN_AVAILABLE

    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Library directory not found: {directory}")

    records = []
    for file in dir_path.rglob("*"):
        if file.suffix.lower() not in AUDIO_EXTS:
            continue

        stem = file.stem
        artist, name = "Unknown", stem
        if " - " in stem:
            parts = stem.split(" - ", 1)
            if parts[0].strip():
                artist = parts[0].strip()
                name = parts[1].strip()

        records.append(
            {
                "Name": name,
                "Artist": artist,
                "Genre": None,
                "Location": str(file.resolve()),
                "Play Count": 0,
                "Skip Count": 0,
                "Year": None,
                "BPM": None,
                "Duration": None,
                "Album": None,
            }
        )

    if not records:
        logging.warning("No audio files found in %s", directory)
        return pd.DataFrame(
            columns=["Name", "Artist", "Genre", "Location",
                     "Play Count", "Skip Count", "Year", "BPM", "Duration", "Album"]
        )

    df = pd.DataFrame(records)
    df.dropna(subset=["Name", "Artist"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Enrich with mutagen tags (fills Name from title tag too if available)
    if mutagen_enabled and MUTAGEN_AVAILABLE:
        df = enrich_dataframe(df, enabled=True)
    elif not mutagen_enabled:
        logging.info("Mutagen tag extraction disabled (MUTAGEN_ENABLED=false).")

    logging.info(
        "Scanned %d audio files from %s  (year: %d, bpm: %d)",
        len(df),
        directory,
        df["Year"].notna().sum() if "Year" in df.columns else 0,
        df["BPM"].notna().sum() if "BPM" in df.columns else 0,
    )
    return df


# ---------------------------------------------------------------------------
# Save slim JSON
# ---------------------------------------------------------------------------


def save_itunes_json(df: pd.DataFrame, path) -> None:
    """Persist a library DataFrame in the slim iTunes JSON format."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {"tracks": df.to_dict(orient="records")}
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logging.info("Saved %d tracks to %s", len(df), p)
