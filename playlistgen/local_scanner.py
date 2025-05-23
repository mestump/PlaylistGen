import os
import logging
import re
from mutagen.mp3 import MP3
from mutagen.flac import FLAC
from mutagen.m4a import M4A
from mutagen.oggvorbis import OggVorbis
from mutagen.wave import WAVE
from mutagen import MutagenError, File
from pathlib import Path

logging.basicConfig(level=logging.INFO)

def parse_year(date_str: str) -> int | None:
    if not date_str:
        return None
    # Simplistic parser: look for a 4-digit number
    match = re.search(r'\d{4}', str(date_str)) # Ensure date_str is a string for regex
    if match:
        year = int(match.group(0))
        # Basic validation for a reasonable year range
        if 1900 <= year <= 2100:
            return year
    return None

def parse_track_number(track_str: str) -> int | None:
    if not track_str:
        return None
    try:
        # Handle cases like "1/12" or just "1"
        return int(str(track_str).split('/')[0])
    except ValueError:
        return None

def scan_local_library(library_root_path: str) -> list[dict]:
    all_tracks = []
    supported_extensions = {".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wav"}
    
    logging.info(f"Starting scan of local library at: {library_root_path}")

    for dirpath, _, filenames in os.walk(library_root_path):
        for filename in filenames:
            file_path = Path(dirpath) / filename
            if file_path.suffix.lower() in supported_extensions:
                track_info = {'Location': str(file_path)}
                try:
                    audio = File(file_path, easy=True)
                    if audio is None:
                        logging.warning(f"Could not process file (None returned): {file_path}")
                        continue
                    
                    # Common tags (easy=True handles many differences)
                    track_info['Name'] = audio.get('title', [None])[0]
                    track_info['Artist'] = audio.get('artist', [None])[0]
                    track_info['Album Artist'] = audio.get('albumartist', [None])[0] # Often more reliable
                    track_info['Album'] = audio.get('album', [None])[0]
                    track_info['Genre'] = audio.get('genre', [None])[0]
                    
                    # Year extraction
                    raw_year_str = None
                    year_keys = ['date', 'originaldate', 'year', 'tdrc'] # Common year tags
                    for key in year_keys:
                        val = audio.get(key, [None])[0]
                        if val:
                            raw_year_str = val
                            break
                    track_info['Year'] = parse_year(raw_year_str)

                    # Track Number
                    raw_track_str = audio.get('tracknumber', [None])[0]
                    track_info['Track Number'] = parse_track_number(raw_track_str)

                    # Total Time (duration)
                    if audio.info:
                        track_info['Total Time'] = int(audio.info.length * 1000)  # in milliseconds
                    else:
                        track_info['Total Time'] = None
                        logging.warning(f"Could not get duration for: {file_path}")

                    # Use Album Artist if Artist is missing
                    if not track_info['Artist'] and track_info['Album Artist']:
                        track_info['Artist'] = track_info['Album Artist']

                    # Ensure all expected keys are present, matching iTunes structure somewhat
                    # (some like 'Persistent ID', 'Track ID' are iTunes specific and not applicable here)
                    for key in ['Name', 'Artist', 'Album', 'Genre', 'Year', 'Track Number', 'Total Time']:
                        if key not in track_info:
                            track_info[key] = None
                            
                    all_tracks.append(track_info)
                    logging.debug(f"Successfully processed: {file_path}")

                except MutagenError as e:
                    logging.warning(f"MutagenError processing {file_path}: {e}")
                except Exception as e:
                    logging.error(f"Unexpected error processing {file_path}: {e}", exc_info=True)
            
    logging.info(f"Scan complete. Found {len(all_tracks)} supported tracks.")
    return all_tracks

if __name__ == '__main__':
    # Example usage:
    # Replace with a path to a small test music directory
    # test_lib_path = "/path/to/your/music/test_folder" 
    # if os.path.exists(test_lib_path):
    #     tracks = scan_local_library(test_lib_path)
    #     for t in tracks:
    #         print(t)
    # else:
    #     print(f"Test library path not found: {test_lib_path}")
    pass
