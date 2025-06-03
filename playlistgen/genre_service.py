"""Genre lookup Service"""

from pathlib import Path
import shelve
import requests
from urllib.parse import quote
from .config import load_config
from typing import Optional

# Cache file for genres
CACHE_PATH = Path.home() / ".playlistgen" / "genre_cache.db"


def fetch_genre_online(artist: str, track: str):
    """
    Fetch all Last.fm tags for a track (not just the first genre).
    Returns a list of tag strings, or [] if none.
    """
    cfg = load_config()
    api_key = cfg.get("LASTFM_API_KEY")
    if not api_key:
        raise ValueError(
            "LASTFM_API_KEY is not set. Please set the environment variable or add it to config.yml"
        )

    key = f"{artist} - {track}".lower()
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with shelve.open(str(CACHE_PATH)) as db:
        if key in db:
            return db[key]

        url = (
            f"https://ws.audioscrobbler.com/2.0/"
            f"?method=track.gettoptags"
            f"&artist={quote(artist)}"
            f"&track={quote(track)}"
            f"&api_key={api_key}&format=json"
        )
        try:
            resp = requests.get(url, timeout=5)
            data = resp.json()
            tags = data.get("toptags", {}).get("tag", [])
            tag_list = [t["name"] for t in tags]
        except Exception:
            tag_list = []

        db[key] = tag_list
        return tag_list


# For compatibility
fetch_genre = fetch_genre_online
