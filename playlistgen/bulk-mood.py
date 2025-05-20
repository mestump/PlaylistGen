import os
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests

# -------------------
# CONFIG
# -------------------
LASTFM_API_URL = "https://ws.audioscrobbler.com/2.0/"
CACHE_FILE = "lastfm_tags_cache.json"
ITUNES_JSON = "./itunes_slimmed.json"          # Edit as needed
SPOTIFY_DIR = "./spotify_history"              # Edit as needed
CONCURRENCY = 8                               # 8 threads is pretty safe for Last.fm
SAVE_EVERY = 250                              # Save cache to disk every N new lookups

def load_lastfm_api_key():
    # Try env variable, then config file
    key = os.environ.get("LASTFM_API_KEY")
    if key:
        return key
    # Try playlistgen config if available
    try:
        from playlistgen.config import load_config
        cfg = load_config()
        return cfg.get("LASTFM_API_KEY")
    except Exception:
        pass
    raise RuntimeError(
        "LASTFM_API_KEY not set. Please set the environment variable or add it to config.yml"
    )

def fetch_lastfm_tags(artist, track, api_key):
    params = {
        "method": "track.gettoptags",
        "artist": artist,
        "track": track,
        "api_key": api_key,
        "format": "json"
    }
    try:
        resp = requests.get(LASTFM_API_URL, params=params, timeout=6)
        tags = [t['name'] for t in resp.json().get('toptags', {}).get('tag', [])]
        return tags
    except Exception as e:
        return []

def get_tracklist_from_itunes_json(path):
    try:
        import pandas as pd
        df = pd.read_json(path)
        if 'tracks' in df.columns:
            df = pd.DataFrame(df['tracks'])
        return set((str(a), str(t)) for a, t in zip(df['Artist'], df['Name']))
    except Exception as e:
        print(f"Failed to parse iTunes JSON: {e}")
        return set()

def get_tracklist_from_spotify_history(directory):
    dir_path = Path(directory)
    track_set = set()
    for json_file in dir_path.rglob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for entry in data:
                    artist = entry.get("master_metadata_album_artist_name")
                    track = entry.get("master_metadata_track_name")
                    if artist and track:
                        track_set.add((artist, track))
        except Exception as e:
            print(f"Failed to load {json_file}: {e}")
    return track_set

def save_cache(cache, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

def bulk_fetch_tags(tracklist, cache_path=CACHE_FILE, concurrency=CONCURRENCY, save_every=SAVE_EVERY):
    api_key = load_lastfm_api_key()
    cache_file = Path(cache_path)
    # Load or initialize cache
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = {}

    to_process = [t for t in tracklist if f"{t[0]} - {t[1]}" not in cache]
    print(f"Processing {len(to_process)} new tracks, {len(cache)} already cached.")

    def worker(pair):
        artist, track = pair
        tags = fetch_lastfm_tags(artist, track, api_key)
        return (f"{artist} - {track}", tags)

    completed = 0
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(worker, pair): pair for pair in to_process}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching tags"):
            key, tags = future.result()
            cache[key] = tags
            completed += 1
            if completed % save_every == 0:
                save_cache(cache, cache_path)
                tqdm.write(f"Checkpoint: {completed} lookups saved.")

    save_cache(cache, cache_path)
    print("Done! Total tracks cached:", len(cache))
    return cache

if __name__ == "__main__":
    # Aggregate all unique (artist, track) pairs
    all_tracks = set()
    all_tracks |= get_tracklist_from_itunes_json(ITUNES_JSON)
    all_tracks |= get_tracklist_from_spotify_history(SPOTIFY_DIR)
    print(f"Total unique (artist, track) pairs: {len(all_tracks)}")

    bulk_fetch_tags(all_tracks)

