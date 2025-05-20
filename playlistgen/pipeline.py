# file: playlistgen/pipeline.py

import logging
from pathlib import Path
from .itunes import convert_itunes_xml, load_itunes_json
from .tag_mood_service import generate_tag_mood_cache, load_tag_mood_db
from .spotify_profile import build_profile, load_profile
from .scoring import score_tracks
from .clustering import cluster_tracks, name_cluster
from .playlist_builder import build_playlists


def ensure_itunes_json(cfg):
    itunes_json = Path(cfg['ITUNES_JSON'])
    itunes_xml = Path(cfg.get("ITUNES_XML", "Itunes Library.xml"))
    if not itunes_json.exists() or (itunes_xml.exists() and itunes_xml.stat().st_mtime > itunes_json.stat().st_mtime):
        logging.info(f"Converting iTunes XML to JSON: {itunes_xml} -> {itunes_json}")
        convert_itunes_xml(str(itunes_xml), str(itunes_json))
    return itunes_json

def ensure_tag_mood_cache(cfg, itunes_json):
    tag_mood_path = Path(cfg['TAG_MOOD_CACHE'])
    spotify_dir = Path(cfg['SPOTIFY_DIR'])
    logging.info(f"Generating Last.fm tag mood cache at {tag_mood_path}")
    generate_tag_mood_cache(itunes_json, spotify_dir, tag_mood_path)
    return tag_mood_path

def generate_profile(cfg, tag_mood_path):
    spotify_dir = Path(cfg['SPOTIFY_DIR'])
    logging.info(f"Building Spotify taste profile from {spotify_dir}")
    build_profile(spotify_dir, tag_mood_path=tag_mood_path)

def run_pipeline(cfg):
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting playlist generation pipeline")

    itunes_json = ensure_itunes_json(cfg)
    tag_mood_path = ensure_tag_mood_cache(cfg, itunes_json)
    generate_profile(cfg, tag_mood_path)

    itunes_df = load_itunes_json(str(itunes_json))
    tag_mood_db = load_tag_mood_db(str(tag_mood_path))
    profile = load_profile(cfg['PROFILE_PATH'])

    logging.info("Scoring tracks")
    scored_df = score_tracks(itunes_df, config=profile, tag_mood_db=tag_mood_db)

    n_clusters = int(cfg.get('CLUSTER_COUNT', 6))
    num_playlists = int(cfg.get('NUM_PLAYLISTS', n_clusters))
    cluster_by_year = cfg.get('YEAR_MIX_ENABLED', False)
    year_range = int(cfg.get('YEAR_MIX_RANGE', 0))
    cluster_by_mood = cfg.get('CLUSTER_BY_MOOD', False)
    min_tracks_per_year = int(cfg.get('MIN_TRACKS_PER_YEAR', 10))

    clusters = cluster_tracks(
        scored_df,
        n_clusters=n_clusters,
        cluster_by_year=cluster_by_year,
        year_range=year_range,
        cluster_by_mood=cluster_by_mood,
        min_tracks_per_year=min_tracks_per_year,
    )

    from random import shuffle
    shuffle(clusters)
    selected_clusters = clusters[:num_playlists]

    for i, playlist in enumerate(selected_clusters):
        label = name_cluster(playlist, i)
        build_playlists([playlist], scored_df, name_fn=lambda *_: label)
        logging.info(f"Playlist {label} built with {len(playlist)} tracks")