import argparse
import logging
from pathlib import Path
from typing import Optional

import joblib

from .playlist_scraper import (
    fetch_spotify_playlists,
    fetch_youtube_playlists,
    fetch_apple_music_playlists,
)
from .pattern_analyzer import analyze_playlists


def train_cluster_model(
    query: str,
    limit: int = 10,
    output: str = "playlist_model.joblib",
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
) -> Path:
    """Fetch playlists online and train a clustering model.

    Returns the path to the saved model file.
    """
    logging.info("Fetching playlists for query: %s", query)
    playlists = fetch_spotify_playlists(query, limit, client_id, client_secret)
    playlists += fetch_youtube_playlists(query, limit)
    playlists += fetch_apple_music_playlists(query, limit)

    if not playlists:
        raise RuntimeError("No playlists fetched; cannot train model")

    logging.info("Analyzing %d playlists", len(playlists))
    result = analyze_playlists(playlists)
    if not result:
        raise RuntimeError("Playlist analysis failed")

    model_data = {"model": result["model"], "vectorizer": result["vectorizer"]}
    out_path = Path(output)
    joblib.dump(model_data, out_path)
    logging.info("Model saved to %s", out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a playlist clustering model from online playlists"
    )
    parser.add_argument("query", help="Search query for playlists, e.g. 'rock'")
    parser.add_argument(
        "--limit", type=int, default=10, help="Number of playlists per service"
    )
    parser.add_argument(
        "--output", default="playlist_model.joblib", help="Path to save the model"
    )
    parser.add_argument("--client-id", help="Spotify client ID")
    parser.add_argument("--client-secret", help="Spotify client secret")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    try:
        train_cluster_model(
            args.query,
            limit=args.limit,
            output=args.output,
            client_id=args.client_id,
            client_secret=args.client_secret,
        )
    except Exception as exc:
        logging.error("Training failed: %s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
