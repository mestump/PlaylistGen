import os
import json
import time
from pathlib import Path
import shelve
import logging
import re
import math
import collections
import pandas as pd # Ensured pandas is imported
# Load config for API keys
from .config import load_config

try:
    import requests
except ImportError:
    requests = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): # type: ignore
        return iterable

# ---- Mood Heuristic Mapping ----
MOODS = {
    "Happy":      ["happy", "feel good", "cheerful", "uplifting", "good mood"],
    "Sad":        ["sad", "melancholy", "melancholic", "heartbreak", "somber"],
    "Angry":      ["angry", "aggressive", "fierce", "rage"],
    "Chill":      ["chill", "chillout", "mellow", "laid back", "relax", "soothing", "calm"],
    "Energetic":  ["energetic", "high energy", "party", "dance", "upbeat", "fast"],
    "Romantic":   ["romantic", "love song", "ballad"],
    "Epic":       ["epic", "anthemic", "dramatic", "orchestral"],
    "Dreamy":     ["dreamy", "ethereal", "ambient", "spacey"],
    "Groovy":     ["groovy", "funky", "swing"],
    "Nostalgic":  ["nostalgia", "retro", "oldies", "classic"],
}
PRIORITY = ["Happy","Sad","Chill","Energetic","Romantic","Epic","Dreamy","Groovy","Nostalgic"]

# ---- Config / Cache ----
_cfg = load_config()
CACHE_PATH = Path(_cfg.get('TAG_MOOD_CACHE'))
SHELVE_PATH = Path(_cfg.get('CACHE_DB'))
# Last.fm API key: environment variable overrides config
API_KEY = os.getenv("LASTFM_API_KEY") or load_config().get("LASTFM_API_KEY")

def canonical_mood(tags, tag_counts=None):
    scores = collections.defaultdict(float)
    for raw in tags:
        clean = re.sub(r"[^\w\s]", " ", raw.lower()).strip()
        for mood, keys in MOODS.items():
            for k in keys:
                if k in clean:
                    # Weight by inverse tag frequency if available (optional)
                    weight = 1.0
                    if tag_counts and clean in tag_counts:
                        count = tag_counts.get(clean, 1)
                        if count <= 1:
                            weight = 1.0
                        else:
                            weight = 1.0 / math.log10(count)
                    scores[mood] += weight
    if scores:
        best = max(scores, key=lambda m: (scores[m], -PRIORITY.index(m) if m in PRIORITY else 99))
        return best
    return None

def fetch_lastfm_tags(artist, track, api_key, cache_db=None):
    """Returns a list of tags from Last.fm for (artist, track)."""
    if not api_key:
        raise RuntimeError(
            "LASTFM_API_KEY is not set. Please set the environment variable or add it to config.yml"
        )
    key = f"{artist.lower()} - {track.lower()}"
    if cache_db is not None:
        if key in cache_db:
            return cache_db[key]
    url = (
        f"https://ws.audioscrobbler.com/2.0/?method=track.gettoptags"
        f"&artist={requests.utils.quote(artist)}"
        f"&track={requests.utils.quote(track)}"
        f"&api_key={api_key}&format=json"
    )
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status() # Raise an exception for HTTP errors
        tags = []
        data = resp.json()
        for tag in data.get("toptags", {}).get("tag", []):
            tname = tag.get("name", "")
            if tname:
                tags.append(tname)
        if cache_db is not None:
            cache_db[key] = tags
        # Optional throttling:
        # time.sleep(0.1)
        return tags
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed for Last.fm tags for {artist} - {track}: {e}")
        if cache_db is not None:
            cache_db[key] = [] # Cache empty list on failure to avoid retrying constantly
        return []
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode failed for Last.fm tags for {artist} - {track}: {e}. Response text: {resp.text[:200] if resp else 'No response'}")
        if cache_db is not None:
            cache_db[key] = []
        return []
    except Exception: # Catch any other unexpected errors
        logging.exception(f"Unexpected error fetching Last.fm tags for {artist} - {track}")
        if cache_db is not None:
            cache_db[key] = []
        return []


def batch_tag_and_mood(track_list, api_key=API_KEY, out_json_path=CACHE_PATH, shelve_path=None):
    """
    For each (artist, track) in track_list:
      - Look up cached tags, else pull from Last.fm
      - Map to mood (using canonical_mood)
      - Save results to disk as JSON and Shelve
    """
    processed = 0
    skipped = 0
    
    output_p = Path(out_json_path) # Use Path object
    output_p.parent.mkdir(parents=True, exist_ok=True)

    if shelve_path is None:
        shelve_p = SHELVE_PATH
    else:
        shelve_p = Path(shelve_path)
    shelve_p.parent.mkdir(parents=True, exist_ok=True)

    # Load existing progress if resuming
    if output_p.exists():
        try:
            with open(output_p, "r", encoding="utf-8") as f:
                mood_db = json.load(f)
        except json.JSONDecodeError:
            logging.error(f"Could not decode existing mood cache at {output_p}. Starting fresh.")
            mood_db = {}
    else:
        mood_db = {}

    tag_counts = collections.Counter()
    with shelve.open(str(shelve_p)) as cache_db: # Ensure shelve path is string
        for artist, track in tqdm(track_list, desc="Fetching tags/moods"):
            artist_str = str(artist) if artist is not None else ""
            track_str = str(track) if track is not None else ""
            
            if not artist_str or not track_str:
                logging.warning(f"Skipping track with missing artist or name: Artist='{artist_str}', Track='{track_str}'")
                skipped +=1
                continue

            track_id = f"{artist_str} - {track_str}".strip().lower()
            # Check if mood specifically is already cached, not just if track_id exists with potentially no mood
            if track_id in mood_db and mood_db[track_id].get('mood') is not None:
                logging.debug(f"Skipping {artist_str} - {track_str}: mood already cached")
                skipped += 1
                continue

            logging.debug(f"Processing {artist_str} - {track_str}")
            processed += 1
            tags = fetch_lastfm_tags(artist_str, track_str, api_key, cache_db)
            for t in tags:
                tag_counts[t.lower()] += 1

            mood = canonical_mood(tags, tag_counts)
            mood_db[track_id] = {"tags": tags, "mood": mood}

            if processed > 0 and processed % 100 == 0:
                try:
                    with open(output_p, "w", encoding="utf-8") as f:
                        json.dump(mood_db, f, indent=2)
                except Exception as e:
                    logging.error(f"Error saving intermediate mood_db to {output_p}: {e}")


        try:
            with open(output_p, "w", encoding="utf-8") as f:
                json.dump(mood_db, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving final mood_db to {output_p}: {e}")
            
    logging.info(f"Done: {len(mood_db)} tracks written to {output_p}")
    logging.info(f"Mood-tagged {processed} tracks; skipped {skipped} tracks (already cached or missing data)")
    return processed, skipped

def load_tag_mood_db(path=CACHE_PATH):
    """Load the tag/mood cache if available, else return empty dict."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logging.error(f"Could not decode mood cache at {p}. Returning empty DB.")
        return {}

# Signature changed as per instructions
def generate_tag_mood_cache(tracks_df: pd.DataFrame, spotify_data_dir_str: str, output_path_str: str):
    """
    Scan all tracks in the provided DataFrame and Spotify directory,
    fetch Last.fm tags for each, and save them to the mood/tag cache as JSON.
    """
    output_p = Path(output_path_str) # Use Path object
    output_p.parent.mkdir(parents=True, exist_ok=True)
    
    tracks_for_batch = []

    # Use tracks_df to prepare tracks_for_batch
    if tracks_df is None or tracks_df.empty:
        logging.warning("Received empty or None DataFrame for mood tagging from primary source.")
        # tracks_for_batch remains empty
    else:
        if 'Artist' not in tracks_df.columns or 'Name' not in tracks_df.columns:
            logging.error("Tracks DataFrame is missing 'Artist' or 'Name' columns for mood tagging.")
            # tracks_for_batch remains empty if essential columns are missing
        else:
            # Create a working copy for modification if necessary
            temp_df = tracks_df[['Artist', 'Name']].copy()
            temp_df.dropna(subset=['Artist', 'Name'], inplace=True)
            # Convert to list of tuples, ensuring values are strings
            tracks_for_batch.extend([
                (str(artist), str(name)) for artist, name in temp_df.to_numpy()
            ])

    # Add all tracks from Spotify play history
    if spotify_data_dir_str: # Check if the string path is provided
        spotify_path = Path(spotify_data_dir_str)
        if spotify_path.exists() and spotify_path.is_dir():
            from glob import glob # Keep import local if only used here
            # import os # os is already imported globally
            logging.info(f"Scanning Spotify data in: {spotify_path}")
            for file_path_str in glob(os.path.join(spotify_data_dir_str, "*.json")):
                try:
                    with open(file_path_str, "r", encoding="utf-8") as f:
                        spotify_entries = json.load(f)
                        for entry in spotify_entries:
                            artist = entry.get('master_metadata_album_artist_name')
                            title = entry.get('master_metadata_track_name')
                            if artist and title:
                                tracks_for_batch.append((str(artist), str(title)))
                except json.JSONDecodeError:
                    logging.error(f"Could not decode JSON from Spotify file: {file_path_str}")
                except Exception as e:
                    logging.error(f"Error processing Spotify file {file_path_str}: {e}")
        else:
            logging.warning(f"Spotify data directory not found or is not a directory: {spotify_data_dir_str}")
    
    if not tracks_for_batch:
        logging.warning("No tracks found from any source to process for mood tagging.")
        # Create an empty cache file to signify completion with no data
        try:
            with open(output_p, "w", encoding="utf-8") as f:
                json.dump({}, f)
        except Exception as e:
            logging.error(f"Could not write empty cache file to {output_p}: {e}")
        return

    # Deduplicate tracks_for_batch
    tracks_for_batch = sorted(list(set(tracks_for_batch)))

    logging.info(f"Fetching tags for {len(tracks_for_batch)} unique tracks. This can take a while...")
    processed, skipped = batch_tag_and_mood(
        tracks_for_batch,
        api_key=API_KEY,
        out_json_path=output_p, # Use Path object
        shelve_path=_cfg.get('CACHE_DB', None) # Allow config to specify shelve path
    )
    logging.info(f"Mood-tagged {processed} tracks; skipped {skipped} tracks from batch.")

    # Sanity check
    if output_p.exists():
        try:
            with open(output_p, "r", encoding="utf-8") as f:
                tag_db = json.load(f)
            logging.info(f"Tag DB now contains {len(tag_db)} tracks. Example entry: {next(iter(tag_db.items())) if tag_db else 'None'}")
        except json.JSONDecodeError:
            logging.error(f"Could not decode final mood cache at {output_p} for sanity check.")
        except Exception as e:
            logging.error(f"Error during sanity check of {output_p}: {e}")

    else:
        logging.error(f"Tag mood DB file {output_p} was NOT written!")
