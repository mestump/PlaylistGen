# 🎧 PlaylistGen — Smart Daily Mix Generator

A command-line Python tool for generating smart, mood-aware, Daily-Mix style playlists using your **iTunes** and **Spotify** listening history. Inspired by Spotify's Daily Mix and iTunes metadata, this tool integrates your play/skip feedback, genres, and online mood classification to curate high-quality playlists with a personal touch.

---

## 🚀 Features

* 📦 **Dependency auto-check and install**
* 🔄 **iTunes XML → JSON converter**
* 🧠 **Spotify taste profile builder**
* 📊 **Daily scoring of tracks using play/skip feedback**
* 🌐 **Online genre lookup** (Last.fm → MusicBrainz)
* 🎨 **Genre normalization & clustering**
* 👥 **Artist diversity enforcement**
* 🧩 **Fallback back-fill logic to meet playlist length**
* 📂 **M3U playlist generation with intelligent naming**
* 💿 **Optional Year-based mix generation**

---

## 🧰 Requirements

* Python 3.8+
* pip packages listed in `requirements.txt` (will auto-install on first run)

---

## 📂 Usage

### Export your iTunes Library

1. Open iTunes (or Music app on macOS).
2. Go to `File` → `Library` → `Export Library`.
3. Save the exported XML file (e.g., `iTunes Music Library.xml`) in the root folder.

### Export your Spotify History

1. Visit [Spotify Privacy](https://www.spotify.com/us/account/privacy/) and request your data.
2. Unzip and place all JSON files in the directory specified by your config (default: `./spotify_history`).

### Generate Playlists (Main Pipeline)

From your project root, run:

```bash
python -m playlistgen
```

Or, if installed as a script:

```bash
playlistgen
```

Main options:

* `recache-moods`: Force a rebuild of the Last.fm mood cache
* `--log-level`: Set log level (DEBUG, INFO, etc)
* `--log-level`: Set log level (DEBUG, INFO, etc)
* `--genre`: Filter mix to only tracks matching the given genre
* `--mood`: Filter mix to only tracks matching the given mood
* `--library-input-path <path>`: Specify the path to your music library. This can be a directory containing audio files, an M3U playlist file (`.m3u`, `.m3u8`), or an iTunes XML library file. This CLI argument overrides `LIBRARY_INPUT_PATH` in your `config.yml`.

Example:

```bash
python -m playlistgen recache-moods
```

Specify library path and filter by genre or mood:

```bash
# Using a directory
python -m playlistgen --library-input-path /path/to/my/music/directory --genre "Rap"

# Using an M3U playlist
python -m playlistgen --library-input-path /path/to/my/playlist.m3u --mood "Epic"

# Using an iTunes XML file (explicitly)
python -m playlistgen --library-input-path /path/to/my/itunes_library.xml --genre "Rap" --mood "Energetic"
```

---

## 🔧 Configuration

Create a `config.yml` in the project root or `~/.playlistgen/config.yml` to override default settings.

Key configuration options:

*   **`LIBRARY_INPUT_PATH`**: Path to your music library. This can be a directory of audio files, an M3U playlist, or an iTunes XML file.
    *   Example (directory): `LIBRARY_INPUT_PATH: /path/to/your/music_folder`
    *   Example (M3U): `LIBRARY_INPUT_PATH: /path/to/your/playlist.m3u`
    *   Example (XML): `LIBRARY_INPUT_PATH: /path/to/your/itunes_library.xml`
*   **`ITUNES_JSON`**: Path where the processed library data (as JSON) will be cached.
*   **`ITUNES_XML`**: (Fallback) Path to your iTunes XML library. Used if `LIBRARY_INPUT_PATH` is not specified via CLI or config.
*   **`SPOTIFY_DIR`**: Directory containing your exported Spotify listening history JSON files.
*   **`PROFILE_PATH`**: Path to save the generated taste profile.
*   **`OUTPUT_DIR`**: Directory where generated M3U playlists will be saved.
*   **`LASTFM_API_KEY`**: Your Last.fm API key (required for online genre/mood detection).
*   Other options: `CLUSTER_COUNT`, `MAX_PER_ARTIST`, `TRACKS_PER_MIX`, `YEAR_MIX_ENABLED`, `YEAR_MIX_RANGE`, etc.

Example `config.yml`:

```yaml
# Specify your library source (choose one type)
LIBRARY_INPUT_PATH: /mnt/music_collection/all_tracks/ # Example for a directory
# LIBRARY_INPUT_PATH: /home/user/playlists/favorites.m3u # Example for an M3U
# LIBRARY_INPUT_PATH: /home/user/data/itunes_export.xml # Example for an XML file

ITUNES_JSON: ./data_cache/processed_library.json # Cache for the processed library
ITUNES_XML: ./data_backup/iTunes Library.xml # Fallback if LIBRARY_INPUT_PATH is not set

SPOTIFY_DIR: ./spotify_extended_history
PROFILE_PATH: ./profiles/main_taste_profile.json
OUTPUT_DIR: ./generated_mixes

LASTFM_API_KEY: your_lastfm_api_key
CLUSTER_COUNT: 8
MAX_PER_ARTIST: 4
TRACKS_PER_MIX: 60
YEAR_MIX_ENABLED: false
```

### Library Source Priority

PlaylistGen determines your music library source with the following priority:

1.  **`--library-input-path` (CLI argument)**: If provided, this path is used directly.
2.  **`LIBRARY_INPUT_PATH` (in `config.yml`)**: If the CLI argument is not provided, this configuration key is used.
3.  **`ITUNES_XML` (in `config.yml`)**: If neither of the above is specified, the system falls back to using the path defined in `ITUNES_XML` (expecting an iTunes XML file).

Ensure the specified path points to a valid directory of audio files, an M3U/M3U8 playlist, or an iTunes XML file.

To use online genre/mood detection, set your **Last.fm API key** in the config.

---

## 🎼 How It Works

1. **iTunes Track Import:** Load and normalize your library’s track and genre metadata.
2. **Spotify Profile Build:** Analyze your play history for play/skip counts, artist/track/year preferences, and mood tags.
3. **Genre Recovery:** Fill missing genres using Last.fm tags.
4. **Track Scoring:** Apply custom scoring using plays, skips, artist and year affinity.
5. **Clustering:** TF-IDF clustering by artist, genre, and track name.
6. **Mix Creation:** Build year-based mixes (optional) and cluster-based mixes named by top genres.
7. **M3U Export:** Save playlists into the `./mixes` folder.

---

## 🧠 Mood Detection (Patience is a Virtue)

* Fetches mood probabilities from Last.fm.
* Initial mood tagging can be very slow for large libraries.

---

## 🧪 Development & Testing

Run tests with:

```bash
pytest
```

---

## ⚠️ Notes

* Ensure Spotify JSON files and iTunes JSON library are placed according to config paths.
* Mood tag caching takes time on first run but speeds up after.
* For most users, just run `python -m playlistgen` to generate playlists after setup.

---

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
See LICENSE for details.
