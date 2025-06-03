import os
import logging
from typing import List, Optional
import pandas as pd

try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
except ImportError:  # pragma: no cover - spotipy is optional for tests
    spotipy = None


def _get_spotify_client(client_id: Optional[str] = None, client_secret: Optional[str] = None):
    """Return an authenticated spotipy client or None if credentials are missing."""
    if spotipy is None:
        logging.warning("spotipy not installed; cannot fetch Spotify playlists")
        return None
    cid = client_id or os.getenv("SPOTIFY_CLIENT_ID")
    secret = client_secret or os.getenv("SPOTIFY_CLIENT_SECRET")
    if not cid or not secret:
        logging.warning("Spotify credentials not configured; skipping playlist fetch")
        return None
    auth = SpotifyClientCredentials(client_id=cid, client_secret=secret)
    return spotipy.Spotify(auth_manager=auth)


def fetch_spotify_playlists(query: str, limit: int = 5, client_id: Optional[str] = None, client_secret: Optional[str] = None) -> List[pd.DataFrame]:
    """Search Spotify for playlists matching the query and return track DataFrames."""
    sp = _get_spotify_client(client_id, client_secret)
    if sp is None:
        return []
    res = sp.search(q=query, type="playlist", limit=limit)
    playlists = []
    for item in res.get("playlists", {}).get("items", []):
        pid = item.get("id")
        tracks = sp.playlist_items(pid)
        rows = []
        for t in tracks.get("items", []):
            tr = t.get("track")
            if not tr:
                continue
            rows.append({
                "Title": tr.get("name"),
                "Artist": ", ".join(a.get("name") for a in tr.get("artists", [])),
                "Album": tr.get("album", {}).get("name"),
                "Genre": None,  # Spotify API does not expose genre at track level
                "Mood": None,
                "Year": str(tr.get("album", {}).get("release_date", ""))[:4],
            })
        playlists.append(pd.DataFrame(rows))
    return playlists


def fetch_youtube_playlists(*_args, **_kwargs) -> List[pd.DataFrame]:
    """Placeholder for YouTube Music scraping."""
    logging.info("YouTube Music scraping is not implemented in this example")
    return []


def fetch_apple_music_playlists(*_args, **_kwargs) -> List[pd.DataFrame]:
    """Placeholder for Apple Music scraping."""
    logging.info("Apple Music scraping is not implemented in this example")
    return []
