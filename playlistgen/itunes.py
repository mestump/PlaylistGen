import plistlib
import json
import pandas as pd
from pathlib import Path
import logging
import datetime
from .utils import sanitize_label
from .config import load_config
from .playlist_builder import save_m3u


def convert_datetimes(obj):
    """
    Recursively convert datetime objects in dicts/lists to ISO format strings.
    """
    if isinstance(obj, dict):
        return {k: convert_datetimes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_datetimes(i) for i in obj]
    elif isinstance(obj, datetime.datetime):
        return obj.isoformat()
    else:
        return obj

def convert_itunes_xml(in_path: str, out_path: str) -> None:
    """
    Convert iTunes Music Library XML to a slimmed JSON format, handling datetime objects.
    """
    with open(in_path, 'rb') as f:
        plist = plistlib.load(f)
    tracks = list(plist.get('Tracks', {}).values())
    # Convert any datetime objects to ISO strings
    tracks = convert_datetimes(tracks)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'tracks': tracks}, f, ensure_ascii=False, indent=2)
    print(f"Converted {len(tracks)} tracks to {out_path}")

def load_itunes_json(path: str) -> pd.DataFrame:
    """
    Load slimmed iTunes JSON into a pandas DataFrame.
    Handles missing or inconsistent column names and normalizes structure.
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    arr = data.get('tracks', data)
    df = pd.DataFrame(arr)

    # Normalize song name column
    for col in ['Name', 'Title', 'Track Name']:
        if col in df.columns:
            df.rename(columns={col: 'Name'}, inplace=True)
            break

    # Only keep the key columns
    col_map = {
        'Artist': 'Artist',
        'Genre': 'Genre',
        'Location': 'Location',
        'Play Count': 'Play Count',
        'Skip Count': 'Skip Count'
    }
    existing = {old: new for old, new in col_map.items() if old in df.columns}
    df = df.rename(columns=existing)
    cols = [c for c in ['Name','Artist','Genre','Location','Play Count','Skip Count'] if c in df.columns]
    df = df[cols]

    # Drop rows missing essential data
    df.dropna(subset=['Name', 'Artist'], inplace=True)

    # Clean up Genre
    if 'Genre' in df.columns:
        df['Genre'] = df['Genre'].fillna('').astype(str).str.strip().str.title()

    # Ensure play/skip counts are ints
    for num in ['Play Count', 'Skip Count']:
        if num in df.columns:
            df[num] = pd.to_numeric(df[num], errors='coerce').fillna(0).astype(int)

    return df
