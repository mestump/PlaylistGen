"""
GPT Playlist Agent — v6.3
========================
Usage:
  python PlaylistAgent.py convert --input "iTunes Music Library.xml" --output itunes_slimmed.json
  python PlaylistAgent.py [--build-profile] [--force-refresh]
"""
from __future__ import annotations
import argparse, subprocess, sys, os, logging, json, pathlib, random, datetime as dt, time, urllib.parse, functools, shelve, plistlib
from typing import Dict, List, Set, Tuple, Optional

# Dependency auto‑install mapping
IMPORT_TO_PIP = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'sklearn': 'scikit-learn',
    'requests': 'requests'
}

def check_deps():
    missing = []
    for mod, pkg in IMPORT_TO_PIP.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing packages: {missing}")
        resp = input("Install now? [y/N] ")
        if resp.lower().startswith('y'):
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
        else:
            sys.exit("Please install dependencies and retry.")

check_deps()

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
try:
    import hdbscan
except ImportError:
    hdbscan = None
import requests

# Configuration
CONFIG: Dict = {
    'ITUNES_JSON': './itunes_slimmed.json',
    'SPOTIFY_DIR': './spotify_history',
    'PROFILE_PATH': './taste_profile.json',
    'OUTPUT_DIR': './mixes',
    'LASTFM_API_KEY': 'XXXX',  # set your Last.fm key here
    'CLUSTER_COUNT': 6,
    'MAX_PER_ARTIST': 5,
    'TRACKS_PER_MIX': 50,
    'YEAR_MIX_ENABLED': True,
    'YEAR_MIX_RANGE': 1,
}

logging.basicConfig(level=logging.INFO, format='%(message)s')

MOOD_TAG_MAP = {
    "chill": ["chill", "relaxing", "ambient", "lo-fi", "downtempo"],
    "sad": ["sad", "melancholy", "depressing", "blue", "emotional"],
    "happy": ["happy", "upbeat", "feel good", "joyful", "sunny"],
    "angry": ["angry", "aggressive", "rage", "hardcore", "intense"],
    "hype": ["party", "dance", "energetic", "club", "high energy"]
}

# AcousticBrainz-based mood tagging fallback
import requests
import time

def get_mbids(artist: str, title: str) -> list[str]:
    """Return a list of candidate MBIDs for the given artist and track."""
    base_url = "https://musicbrainz.org/ws/2/recording/"
    query = f'recording:"{title}" AND artist:"{artist}"'
    params = {
        "query": query,
        "fmt": "json",
        "limit": 3
    }
    headers = {
        "User-Agent": "PlaylistAgent/1.0 (your_email@example.com)"
    }
    try:
        resp = requests.get(base_url, params=params, headers=headers)
        data = resp.json()
        mbids = [r['id'] for r in data.get('recordings', [])]
        return mbids
    except Exception as e:
        print(f"[MBID] Failed for {artist} - {title}: {e}")
        return []

def fetch_acousticbrainz_mood(mbid: str) -> dict[str, float]:
    """Fetch mood probabilities from AcousticBrainz for a given MBID."""
    url = f"https://acousticbrainz.org/{mbid}/high-level"
    try:
        resp = requests.get(url)
        data = resp.json()
        highlevel = data.get('highlevel', {})
        moods = {}
        for key in highlevel:
            if key.startswith('mood_') and 'value' in highlevel[key]:
                mood = key.replace("mood_", "")
                val = highlevel[key]['value']
                prob = highlevel[key].get('probability', 0)
                if not val.startswith("not_"):
                    moods[mood] = prob
        return moods
    except Exception as e:
        print(f"[Mood] Error fetching {mbid}: {e}")
        return {}

def get_mood_tags(artist: str, title: str) -> dict[str, float]:
    """End-to-end lookup: artist + title -> MBID -> mood probabilities, with caching"""
    import shelve
    key = f"{artist} - {title}".lower()
    with shelve.open("mood_cache") as cache:
        if key in cache:
            return cache[key]
        time.sleep(1)
        mbids = get_mbids(artist, title)
        for mbid in mbids:
            moods = fetch_acousticbrainz_mood(mbid)
            if moods:
                cache[key] = moods
                return moods
        cache[key] = {}
        return {}

# iTunes XML -> JSON converter
def convert_itunes_xml(input_path: str, output_path: str):
    plist = plistlib.load(open(input_path, 'rb'))
    tracks = list(plist.get('Tracks', {}).values())
    json.dump({'tracks': tracks}, open(output_path, 'w', encoding='utf-8'), ensure_ascii=False)
    print(f"Converted {len(tracks)} tracks to {output_path}")

# Load iTunes JSON
def load_itunes(path: str) -> pd.DataFrame:
    data = json.load(open(path, 'r', encoding='utf-8'))
    arr = data['tracks'] if isinstance(data, dict) and 'tracks' in data else data
    df = pd.DataFrame(arr)
    for col in ['Name','Title','Track Name']:
        if col in df.columns:
            df.rename(columns={col: 'Name'}, inplace=True)
            break
    df = df.rename(columns={'Artist':'Artist','Genre':'Genre','Location':'Location','Play Count':'Play Count','Skip Count':'Skip Count'})
    df = df[['Name','Artist','Genre','Location','Play Count','Skip Count']]
    df.dropna(subset=['Name','Artist'], inplace=True)
    df['Genre'] = df['Genre'].fillna('').astype(str).str.strip().str.title()
    return df

# Build Spotify profile with verbose and correct parsing

def build_profile(sp_dir: pathlib.Path) -> Dict:
    logging.info(f"Scanning directory: {sp_dir.resolve()}")
    files = list(sp_dir.rglob('*.json'))
    if not files:
        logging.warning("No Spotify JSON files found. Check your CONFIG['SPOTIFY_DIR'] path.")

    artist_scores: Dict[str, int] = {}
    year_scores: Dict[int, int] = {}
    track_play_counts: Dict[str, int] = {}
    skip_counts: Dict[str, int] = {}
    track_moods: Dict[str, List[str]] = {}

    track_index = 0
    for f in files:
        logging.info(f"Processing: {f.name}")
        try:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
        except Exception as e:
            logging.warning(f"Failed to load {f.name}: {e}")
            continue

        for entry in data:
            artist = entry.get("master_metadata_album_artist_name")
            track = entry.get("master_metadata_track_name")
            album = entry.get("master_metadata_album_album_name")
            ms_played = entry.get("ms_played", 0)
            skipped = entry.get("skipped", False)
            ts = entry.get("ts")

            if not artist or not track:
                continue

            artist_scores[artist] = artist_scores.get(artist, 0) + ms_played
            track_id = f"{artist} - {track}"
            track_play_counts[track_id] = track_play_counts.get(track_id, 0) + 1
            if skipped:
                skip_counts[track_id] = skip_counts.get(track_id, 0) + 1

            try:
                year = dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).year
                year_scores[year] = year_scores.get(year, 0) + 1
            except:
                continue

            if track_id not in track_moods:
                logging.info(f"Looking up mood for: {track_id}")
                moods = get_mood_tags(artist, track)
                if moods:
                   logging.info(f"→ Moods found: {list(moods.keys())}")
                   track_moods[track_id] = moods
                else:
                   logging.info(f"→ No mood tags found for: {track_id}")


            track_index += 1
            if track_index % 50 == 0:
                logging.info(f"Status: {track_index} tracks processed so far...")

    profile = {
        "artist_scores": dict(sorted(artist_scores.items(), key=lambda x: -x[1])),
        "year_scores": dict(sorted(year_scores.items())),
        "track_play_counts": track_play_counts,
        "track_skip_counts": skip_counts,
        "track_moods": track_moods,
        "generated_at": dt.datetime.utcnow().isoformat() + "Z"
    }
    logging.info(f"Parsed {len(artist_scores)} artists, {len(year_scores)} years, {len(track_moods)} tracks with moods")
    json.dump(profile, open(CONFIG['PROFILE_PATH'], 'w'), indent=2)
    return profile

# Load cached profile
def load_profile() -> Optional[Dict]:
    if os.path.exists(CONFIG['PROFILE_PATH']):
        return json.load(open(CONFIG['PROFILE_PATH'], 'r'))
    return None

# Online genre lookup
CACHE_DB = pathlib.Path('genre_cache.db')
@functools.lru_cache(maxsize=10000)
def fetch_genre_online(artist: str, track: str) -> Optional[str]:
    import requests, shelve
    key = f"{artist} - {track}".lower()
    try:
        with shelve.open("genres") as db:
            if key in db:
                return db[key]
            url = (
                f"https://ws.audioscrobbler.com/2.0/?method=track.gettoptags"
                f"&artist={requests.utils.quote(artist)}"
                f"&track={requests.utils.quote(track)}"
                f"&api_key={CONFIG['LASTFM_API_KEY']}&format=json"
            )
            data = requests.get(url).json()
            tags = data.get("toptags", {}).get("tag", [])
            if tags:
                tag = tags[0]['name'].title()
                db[key] = tag  # ✅ write inside 'with'
                return tag
    except Exception as e:
        print(f"Genre fetch error for {artist} - {track}: {e}")
    return None

# Clustering & sampling
vec = None
km = None

def cluster_tracks(df: pd.DataFrame) -> List[pd.DataFrame]:
    global vec, km
    df['text_blob'] = df['Artist'] + ' ' + df['Name'] + ' ' + df['Genre']
    vec = TfidfVectorizer()
    X = vec.fit_transform(df['text_blob'].fillna(''))
    if CONFIG['CLUSTER_COUNT'] and not hdbscan:
        km = KMeans(n_clusters=CONFIG['CLUSTER_COUNT'], random_state=0)
        labels = km.fit_predict(X)
    else:
        labels = hdbscan.HDBSCAN(min_cluster_size=10).fit_predict(X)
    df['Cluster'] = labels
    return [df[df['Cluster'] == i] for i in sorted(set(labels))]


def cap_artist(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby('Artist', group_keys=False).head(CONFIG['MAX_PER_ARTIST'])


def fill_short_pool(df: pd.DataFrame, global_df: pd.DataFrame) -> pd.DataFrame:
    need = CONFIG['TRACKS_PER_MIX'] - len(df)
    if need <= 0: return df
    filler = global_df.drop(df.index).head(need)
    return pd.concat([df, filler], ignore_index=True)

# Naming & saving

def sanitize_label(label: str) -> str:
    lbl = label.replace('/', ' - ').replace('\\', ' - ')
    for ch in '<>:"|?*':
        lbl = lbl.replace(ch, '')
    lbl = ' '.join(lbl.split())
    lbl = lbl.rstrip('& ').strip()
    return lbl


def name_cluster(df: pd.DataFrame, *, year_target: int=None) -> str:
    if year_target:
        return f"{year_target} Vibes"
    counts = df['Genre'].value_counts()
    total = len(df)
    top, cnt = counts.idxmax(), counts.max()
    if cnt/total >= 0.8:
        return top
    majors = list(dict.fromkeys(
        g for g in counts.index if counts[g]/total >= 0.2
    ))[:2]
    return ' & '.join(majors) if majors else 'Eclectic'


def save_m3u(df: pd.DataFrame, label: str):
    os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
    date = dt.datetime.now().strftime('%Y-%m-%d')
    safe = sanitize_label(label)
    fname = f"{safe} [{date}].m3u"
    path = os.path.join(CONFIG['OUTPUT_DIR'], fname)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('#EXTM3U\n')
        for _, r in df.iterrows():
            f.write(f"#EXTINF:-1,{r['Artist']} - {r['Name']}\n")
            f.write(f"{r['Location']}\n")
    logging.info(f"Saved {path}")


def extract_year(row) -> Optional[int]:
    # Attempt to extract year from known tags
    loc = row.get('Location', '')
    try:
        parts = loc.split('/')
        for p in parts:
            if p.isdigit() and 1900 < int(p) < 2100:
                return int(p)
    except:
        return None
    return None

def score_track(row, profile) -> float:
    artist_score = profile.get('artist_scores', {}).get(row['Artist'], 0)
    year = extract_year(row)
    year_score = profile.get('year_scores', {}).get(year, 0) if year else 0
    play_bonus = row.get('Play Count', 0)
    skip_penalty = row.get('Skip Count', 0)
    return artist_score + year_score + (2 * play_bonus) - (3 * skip_penalty)

def filter_by_random_year(df: pd.DataFrame, profile: Dict) -> Tuple[Optional[int], pd.DataFrame]:
    years = list(profile.get('year_scores', {}).keys())
    if not years:
        return None, df
    target = random.choice(years)
    lo, hi = target - CONFIG['YEAR_MIX_RANGE'], target + CONFIG['YEAR_MIX_RANGE']
    mask = df['Location'].fillna('').apply(lambda x: any(str(y) in x for y in range(lo, hi+1)))
    return target, df[mask]


# Main orchestration

def run_agent(build_profile_only=False, force_refresh=False):
    start = dt.datetime.now()
    profile = load_profile() or {}
    if not profile or force_refresh:
        logging.info('Building taste profile...')
        profile = build_profile(pathlib.Path(CONFIG['SPOTIFY_DIR']))
        if build_profile_only:
            return
    itunes = load_itunes(CONFIG['ITUNES_JSON'])
    logging.info(f"Loaded {len(itunes)} tracks")
    # Fill missing genres
    for i, r in itunes[itunes['Genre']==''].iterrows():
        itunes.at[i,'Genre'] = fetch_genre_online(r['Artist'], r['Name']) or ''
# Compute scores
    itunes['Score'] = itunes.apply(lambda r: score_track(r, profile), axis=1)

    # Try a year-based mix first
    year_target, year_df = filter_by_random_year(itunes, profile) if CONFIG['YEAR_MIX_ENABLED'] else (None, None)
    if year_target and not year_df.empty:
        year_df = year_df.copy()
        year_df['Score'] = year_df.apply(lambda r: score_track(r, profile), axis=1)
        year_df = year_df.sort_values('Score', ascending=False)
        year_df = cap_artist(year_df)
        year_df = fill_short_pool(year_df, itunes)
        save_m3u(year_df.sample(CONFIG['TRACKS_PER_MIX'], random_state=0), name_cluster(year_df, year_target=year_target))

    # Cluster and build standard mixes
    clusters = cluster_tracks(itunes)
    for cdf in clusters:
        cdf = cdf.copy()
        cdf['Score'] = cdf.apply(lambda r: score_track(r, profile), axis=1)
        cdf = cdf.sort_values('Score', ascending=False)
        cdf = cap_artist(cdf)
        cdf = fill_short_pool(cdf, itunes)
        label = name_cluster(cdf)
        save_m3u(cdf.sample(CONFIG['TRACKS_PER_MIX'], random_state=0), label)
    end = dt.datetime.now()
    logging.info(f"Completed in {(end-start).total_seconds():.1f}s")

# CLI

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')
    conv = sub.add_parser('convert')
    conv.add_argument('--input', required=True)
    conv.add_argument('--output', required=True)
    parser.add_argument('--build-profile', action='store_true')
    parser.add_argument('--force-refresh', action='store_true')
    args = parser.parse_args()
    if args.cmd == 'convert':
        convert_itunes_xml(args.input, args.output)
    else:
        run_agent(build_profile_only=args.build_profile, force_refresh=args.force_refresh)

# EOF

