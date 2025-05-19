import requests
import time
import shelve
import logging

from tqdm import tqdm


TAG_TO_MOOD = {
    'chill': 'chill',
    'relax': 'chill',
    'ambient': 'chill',
    'sad': 'sad',
    'melancholy': 'sad',
    'depressing': 'sad',
    'happy': 'happy',
    'upbeat': 'happy',
    'joyful': 'happy',
    'party': 'hype',
    'dance': 'hype',
    'energetic': 'hype',
    'aggressive': 'angry',
    'hardcore': 'angry',
    'rage': 'angry',
    'relaxed': 'relaxed',
    'blue': 'sad',
    'emotional': 'sad',
    'intense': 'angry',
    # Add more as desired
}

def map_tags_to_mood(tags):
    if not tags:
        return {}
    tags = [t.lower() for t in tags]
    for tag in tags:
        for keyword, mood in TAG_TO_MOOD.items():
            if keyword in tag:
                return {mood: 1.0}
    return {}

def get_mbids(artist: str, title: str) -> list:
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
        resp.raise_for_status()
        data = resp.json()
        mbids = [r['id'] for r in data.get('recordings', [])]
        logging.info(f"MBID lookup: {artist} - {title}: {mbids}")
        return mbids
    except Exception as e:
        logging.error(f"[MBID] Failed for {artist} - {title}: {e}")
        return []

def fetch_acousticbrainz_mood(mbid: str) -> dict:
    url = f"https://acousticbrainz.org/{mbid}/high-level"
    try:
        resp = requests.get(url)
        resp.raise_for_status()
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
        logging.info(f"Fetched moods for {mbid}: {moods}")
        return moods
    except Exception as e:
        logging.error(f"[Mood] Error fetching {mbid}: {e}")
        return {}

def get_mood_tags(artist, title):
    """Try to fetch mood from AcousticBrainz/MBID/cached; return {} if not found."""
    key = f"{artist} - {title}".lower()
    with shelve.open("mood_cache") as cache:
        if key in cache:
            logging.info(f"Cache hit for {key}")
            return cache[key]
        time.sleep(1)
        mbids = get_mbids(artist, title)
        for mbid in mbids:
            moods = fetch_acousticbrainz_mood(mbid)
            if moods:
                cache[key] = moods
                return moods
        cache[key] = {}
        logging.warning(f"No moods found for {key}")
        return {}

# --- This is the function you want to use everywhere else ---
def get_track_mood(artist, title, fetch_genre_online=None):
    """Fetch mood from AcousticBrainz if possible, otherwise fall back to Last.fm top tags."""
    moods = get_mood_tags(artist, title)
    if moods:
        return moods

    # Fallback to Last.fm tags, if fetch_genre_online is provided
    if fetch_genre_online is not None:
        tags = fetch_genre_online(artist, title)
        if tags:
            if isinstance(tags, str):
                tags = [tags]
            mood = map_tags_to_mood(tags)
            if mood:
                return mood

    return {}
