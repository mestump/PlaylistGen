import pandas as pd
import logging
from .spotify_profile import load_profile
from .tag_mood_service import load_tag_mood_db
from .config import load_config

logging.basicConfig(level=logging.INFO)

def score_tracks(itunes_df: pd.DataFrame, config=None, tag_mood_db=None, weights=None):
    """
    Add a 'Score' column to the iTunes DataFrame based on user profile, plays/skips, mood/genre, etc.

    - itunes_df: DataFrame of iTunes tracks
    - config: dict of configuration (optional; may include PROFILE_PATH and TAG_MOOD_CACHE)
    - tag_mood_db: dict mapping 'artist - track' to tags/mood (optional; will load if not provided)
    - weights: dict, weights for each factor
    """
    # Determine config and profile
    if isinstance(config, dict) and 'PROFILE_PATH' in config:
        cfg = config
        profile = load_profile(cfg['PROFILE_PATH'])
    else:
        cfg = load_config()
        profile = config if isinstance(config, dict) else load_profile(cfg['PROFILE_PATH'])
    # Load tag/mood database if not provided
    if tag_mood_db is None:
        tag_mood_db = load_tag_mood_db(cfg.get('TAG_MOOD_CACHE', None))
    # Default weights
    if weights is None:
        weights = {
            'artist': 2.0,
            'genre': 1.0,
            'mood': 1.0,
            'year': 0.5,
            'play': 2.0,
            'skip': -3.0,
        }

    def get_track_id(row):
        return f"{row['Artist']} - {row['Name']}".strip().lower()

    def score_row(row):
        track_id = get_track_id(row)
        artist = row['Artist']
        genre = row['Genre'] if 'Genre' in row and pd.notnull(row['Genre']) else ''
        play_count = row['Play Count'] if 'Play Count' in row else 0
        skip_count = row['Skip Count'] if 'Skip Count' in row else 0

        tag_mood = tag_mood_db.get(track_id, {})
        mood = tag_mood.get('mood')
        tags = tag_mood.get('tags', [])

        # Taste profile scores
        # Taste profile scores with safe defaults
        artist_score = profile.get('artist_scores', {}).get(artist, 0)
        genre_score = profile.get('genre_scores', {}).get(genre.lower(), 0) if genre else 0
        mood_score = profile.get('mood_scores', {}).get(mood, 0) if mood else 0

        # Year scoring (optional)
        year_score = 0
        year = None
        if 'Location' in row and isinstance(row['Location'], str):
            try:
                # Try to extract year from file path, e.g. ".../2004/..."
                for part in row['Location'].split('/'):
                    if part.isdigit() and 1900 < int(part) < 2100:
                        year = int(part)
                        break
            except Exception:
                year = None
        if year:
            year_score = profile.get('year_scores', {}).get(year, 0)

        # Spotify play/skip for track
        spotify_play = profile.get('track_play_counts', {}).get(track_id, 0)
        spotify_skip = profile.get('track_skip_counts', {}).get(track_id, 0)

        # Final scoring formula
        score = (
            weights['artist'] * artist_score +
            weights['genre'] * genre_score +
            weights['mood'] * mood_score +
            weights['year'] * year_score +
            weights['play'] * (play_count + spotify_play) +
            weights['skip'] * (skip_count + spotify_skip)
        )
        return score

    logging.info("Scoring tracks...")
    itunes_df = itunes_df.copy()
    itunes_df['Score'] = itunes_df.apply(score_row, axis=1)
    scored_count = (itunes_df['Score'] > 0).sum()
    zero_score_count = (itunes_df['Score'] == 0).sum()
    logging.info(f'Scoring complete: {scored_count} tracks scored >0, {zero_score_count} zero, {len(itunes_df)-scored_count-zero_score_count} missing, of             {len(itunes_df)} total.')
    if zero_score_count > len(itunes_df)*0.3:
        logging.warning('More than 30% of tracks scored as zero! Check tag/mood/genre mapping.')

    # Ensure 'Mood' column is present and populated for all tracks
    itunes_df['track_id'] = itunes_df.apply(get_track_id, axis=1)
    itunes_df['Mood'] = (
        itunes_df['track_id']
        .map(lambda tid: tag_mood_db.get(tid, {}).get('mood'))
        .fillna('Unknown')
    )
    itunes_df.drop(columns=['track_id'], inplace=True)

    return itunes_df

# Optional: Utility to get top N tracks for debugging
def top_tracks(df, n=10):
    return df.sort_values('Score', ascending=False).head(n)
