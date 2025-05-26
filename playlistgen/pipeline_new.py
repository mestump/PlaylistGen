# file: playlistgen/pipeline_new.py

import logging
from pathlib import Path
import json 
import pandas as pd 
import os 

from .itunes import convert_itunes_xml 
from .tag_mood_service import generate_tag_mood_cache, load_tag_mood_db
from .spotify_profile import build_profile, load_profile
from .scoring import score_tracks
from .clustering import cluster_tracks, name_cluster
from .playlist_builder import build_playlists
from . import library_scanner 

def ensure_library_json(cfg):
    library_input_path_str = cfg.get('LIBRARY_INPUT_PATH')
    library_json_cache_path = Path(cfg['ITUNES_JSON']) 

    input_path = None
    input_type = None 

    if library_input_path_str:
        library_input_path = Path(library_input_path_str)
        if library_input_path.is_dir():
            input_path = library_input_path
            input_type = 'directory'
        elif library_input_path.is_file():
            if library_input_path.suffix.lower() == '.xml':
                input_path = library_input_path
                input_type = 'xml'
            elif library_input_path.suffix.lower() in ['.m3u', '.m3u8']:
                input_path = library_input_path
                input_type = 'm3u'
            else:
                logging.warning(f"Unsupported file type for LIBRARY_INPUT_PATH: {library_input_path_str}")
        else:
            logging.warning(f"LIBRARY_INPUT_PATH is not a valid file or directory: {library_input_path_str}")
    else: 
        itunes_xml_path_str = cfg.get("ITUNES_XML", "Itunes Library.xml") 
        itunes_xml_path = Path(itunes_xml_path_str)
        if itunes_xml_path.exists():
            input_path = itunes_xml_path
            input_type = 'xml'
        else:
            if library_json_cache_path.exists():
                logging.info(f"No primary input specified, but using existing cache: {library_json_cache_path}")
                return library_json_cache_path
            else:
                logging.error(f"No valid library input found. LIBRARY_INPUT_PATH is not set and ITUNES_XML not found at {itunes_xml_path_str}. No cache exists.")
                raise FileNotFoundError("No library input specified (LIBRARY_INPUT_PATH or ITUNES_XML) and no existing cache found.")

    rebuild_cache = False
    if not library_json_cache_path.exists():
        rebuild_cache = True
    elif input_path and input_path.exists(): 
        try:
            if input_type in ['xml', 'm3u']: 
                 if input_path.stat().st_mtime > library_json_cache_path.stat().st_mtime:
                    rebuild_cache = True
            elif input_type == 'directory':
                if input_path.stat().st_mtime > library_json_cache_path.stat().st_mtime:
                     rebuild_cache = True
        except FileNotFoundError:
            rebuild_cache = True

    if rebuild_cache and input_path and input_type:
        logging.info(f"Processing library input: {input_path} ({input_type}) -> {library_json_cache_path}")
        library_json_cache_path.parent.mkdir(parents=True, exist_ok=True) 

        if input_type == 'xml':
            convert_itunes_xml(str(input_path), str(library_json_cache_path))
        else:
            df = None
            if input_type == 'directory':
                df = library_scanner.scan_directory(str(input_path))
            elif input_type == 'm3u':
                df = library_scanner.scan_m3u(str(input_path))

            if df is not None and not df.empty:
                tracks_data = df.to_dict(orient='records')
                with open(library_json_cache_path, 'w', encoding='utf-8') as f:
                    json.dump({'tracks': tracks_data}, f, ensure_ascii=False, indent=2)
                logging.info(f"Saved {len(tracks_data)} tracks to {library_json_cache_path}")
            elif df is not None and df.empty:
                logging.warning(f"Scan of {input_type} from {input_path} resulted in an empty dataset. Cache file will reflect this.")
                with open(library_json_cache_path, 'w', encoding='utf-8') as f:
                    json.dump({'tracks': []}, f, ensure_ascii=False, indent=2)
            else: 
                logging.error(f"Failed to scan {input_type} from {input_path}. Cache not updated.")
                if not library_json_cache_path.exists(): 
                     raise FileNotFoundError(f"Failed to process library and no cache available at {library_json_cache_path}")
    elif not library_json_cache_path.exists():
        logging.error(f"Library cache {library_json_cache_path} does not exist and conditions to rebuild it were not met (e.g. invalid input path).")
        raise FileNotFoundError(f"Library cache does not exist and could not be built: {library_json_cache_path}")
    else:
        logging.info(f"Using existing library cache: {library_json_cache_path}")

    return library_json_cache_path

def load_library_json(path: str) -> pd.DataFrame:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.error(f"Library JSON file not found at {path}")
        return pd.DataFrame(columns=['Name', 'Artist', 'Genre', 'Location', 'Play Count', 'Skip Count'])
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {path}")
        return pd.DataFrame(columns=['Name', 'Artist', 'Genre', 'Location', 'Play Count', 'Skip Count'])
        
    arr = data.get('tracks', data) 
    if not isinstance(arr, list): 
        logging.error(f"JSON structure error in {path}: 'tracks' should be a list, or root should be a list.")
        return pd.DataFrame(columns=['Name', 'Artist', 'Genre', 'Location', 'Play Count', 'Skip Count'])
        
    df = pd.DataFrame(arr)

    title_cols = ['Name', 'Title', 'Track Name', 'Track', 'Song']
    normalized_name = False
    for col in title_cols:
        if col in df.columns:
            df.rename(columns={col: 'Name'}, inplace=True)
            normalized_name = True
            break
    if not normalized_name and 'Name' not in df.columns:
        df['Name'] = None

    artist_cols = ['Artist', 'Album Artist', 'Performer']
    normalized_artist = False
    for col in artist_cols:
        if col in df.columns:
            df.rename(columns={col: 'Artist'}, inplace=True)
            normalized_artist = True
            break
    if not normalized_artist and 'Artist' not in df.columns:
        df['Artist'] = None
        
    if 'Location' not in df.columns:
        df['Location'] = None 

    if 'Genre' not in df.columns:
        df['Genre'] = '' 
    df['Genre'] = df['Genre'].fillna('').astype(str).str.strip().str.title()

    for num_col in ['Play Count', 'Skip Count']:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors='coerce').fillna(0).astype(int)
        else:
            df[num_col] = 0 

    standard_cols = ['Name', 'Artist', 'Genre', 'Location', 'Play Count', 'Skip Count']
    
    for std_col in standard_cols:
        if std_col not in df.columns:
            df[std_col] = None if std_col not in ['Genre', 'Play Count', 'Skip Count'] else ('' if std_col == 'Genre' else 0)

    df = df[standard_cols] 

    if 'Name' in df.columns and 'Artist' in df.columns and normalized_name and normalized_artist :
        df.dropna(subset=['Name', 'Artist'], how='any', inplace=True)
        
    return df

def ensure_tag_mood_cache(cfg, library_json_path): 
    tag_mood_path = Path(cfg['TAG_MOOD_CACHE'])
    spotify_dir = Path(cfg['SPOTIFY_DIR'])
    
    rebuild_mood_cache = True 
    if tag_mood_path.exists() and library_json_path.exists():
        try:
            if library_json_path.stat().st_mtime < tag_mood_path.stat().st_mtime:
                rebuild_mood_cache = False 
        except FileNotFoundError: 
            pass 
    
    if rebuild_mood_cache:
        logging.info(f"Generating Last.fm tag mood cache at {tag_mood_path} using library {library_json_path}")
        tag_mood_path.parent.mkdir(parents=True, exist_ok=True)
        generate_tag_mood_cache(library_json_path, spotify_dir, tag_mood_path) 
    else:
        logging.info(f"Using existing Last.fm tag mood cache: {tag_mood_path}")
    return tag_mood_path

def generate_profile(cfg, tag_mood_path):
    spotify_dir = Path(cfg['SPOTIFY_DIR'])
    profile_path = Path(cfg['PROFILE_PATH']) 
    profile_path.parent.mkdir(parents=True, exist_ok=True) 
    logging.info(f"Building Spotify taste profile from {spotify_dir} to {profile_path}")
    build_profile(spotify_dir, tag_mood_path=tag_mood_path, out_path=profile_path) 

def run_pipeline(cfg, genre=None, mood=None):
    if not logging.getLogger().handlers:
        log_level_str = cfg.get('LOG_LEVEL', 'INFO').upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        logging.basicConfig(level=log_level)
        
    logging.info("Starting playlist generation pipeline")

    library_json_path = ensure_library_json(cfg) 
    tag_mood_path = ensure_tag_mood_cache(cfg, library_json_path) 
    
    profile_path = Path(cfg['PROFILE_PATH'])
    should_generate_profile = True 
    if profile_path.exists() and tag_mood_path.exists():
        try: 
            if profile_path.stat().st_mtime > tag_mood_path.stat().st_mtime: 
                should_generate_profile = False 
        except FileNotFoundError:
            pass 
    
    if should_generate_profile:
        logging.info(f"Generating taste profile at {profile_path}")
        generate_profile(cfg, tag_mood_path) 
    else:
        logging.info(f"Using existing taste profile: {profile_path}")

    library_df = load_library_json(str(library_json_path)) 
    if library_df.empty:
        logging.error("Library data is empty after loading. Cannot proceed.")
        return

    tag_mood_db = load_tag_mood_db(str(tag_mood_path))
    profile = load_profile(cfg['PROFILE_PATH'])

    logging.info("Scoring tracks")
    scored_df = score_tracks(library_df, config=profile, tag_mood_db=tag_mood_db, app_config=cfg)

    if genre or mood:
        filt_df = scored_df.copy() 
        if genre:
            logging.info(f"Filtering tracks by genre: {genre}")
            if 'Genre' in filt_df.columns:
                filt_df = filt_df[
                    filt_df['Genre'].notna() &
                    (filt_df['Genre'].str.lower() == genre.lower())
                ]
            else:
                logging.warning("'Genre' column not found, cannot filter by genre.")
        if mood:
            logging.info(f"Filtering tracks by mood: {mood}")
            if 'Mood' in filt_df.columns: 
                 filt_df = filt_df[
                    filt_df['Mood'].notna() &
                    (filt_df['Mood'].str.lower() == mood.lower())
                ]
            else:
                logging.warning("Mood column not found in scored tracks, cannot filter by mood.")

        if filt_df.empty:
            logging.warning(f"No tracks found matching genre={genre!r} mood={mood!r}")
            return

        parts = []
        if mood: 
            if 'Mood' in filt_df.columns and filt_df['Mood'].str.lower().eq(mood.lower()).any():
                 parts.append(mood.capitalize())
        if genre: 
            if 'Genre' in filt_df.columns and filt_df['Genre'].str.lower().eq(genre.lower()).any():
                parts.append(genre.capitalize())
        
        label_prefix = " & ".join(parts) if parts else "Custom"
        label = f"{label_prefix} Mix"
        
        output_dir = Path(cfg.get('OUTPUT_DIR', './mixes'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        build_playlists(playlist_dfs=[filt_df], all_tracks_df=scored_df, name_fn=lambda *_: label, config=cfg, out_dir=output_dir)
        return

    n_clusters = int(cfg.get('CLUSTER_COUNT', 6))
    
    clusters = cluster_tracks(
        scored_df,
        n_clusters=n_clusters, 
        cluster_by_year=cfg.get('YEAR_MIX_ENABLED', False),
        year_range=int(cfg.get('YEAR_MIX_RANGE', 0)),
        cluster_by_mood=cfg.get('CLUSTER_BY_MOOD', False), 
        min_tracks_per_year=int(cfg.get('MIN_TRACKS_PER_YEAR', 10)),
    )
    
    if not clusters: 
        logging.warning("No clusters were generated. Cannot build playlists.")
        return

    from random import shuffle
    shuffle(clusters) 
    
    num_playlists_to_generate = int(cfg.get('NUM_PLAYLISTS', len(clusters))) 
    selected_clusters = clusters[:num_playlists_to_generate]

    output_dir = Path(cfg.get('OUTPUT_DIR', './mixes'))
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, playlist_df in enumerate(selected_clusters):
        if playlist_df.empty:
            logging.warning(f"Cluster {i+1} is empty, skipping playlist generation for it.")
            continue
        label = name_cluster(playlist_df, i) 
        build_playlists(playlist_dfs=[playlist_df], all_tracks_df=scored_df, name_fn=lambda *_: label, config=cfg, out_dir=output_dir)
