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
* 📁 **Manual library scanning with `--library-dir` flag**
* 🖥️ **Clean progress bars for status updates**

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

If you manage your music files manually and don't use iTunes, skip the steps above and run the tool with `--library-dir /path/to/music`.

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

Running without arguments now launches an interactive menu where you can manage
API keys, log into Spotify, recache metadata, or generate playlists.

Main options (for non-interactive usage):

* `recache-moods`: Force a rebuild of the Last.fm mood cache
* `--log-level`: Set log level (DEBUG, INFO, etc)
* `--genre`: Filter mix to only tracks matching the given genre
* `--mood`: Filter mix to only tracks matching the given mood
* `--library-dir`: Scan a manual music directory instead of an iTunes library

Example:

```bash
python -m playlistgen recache-moods
```

Filter by genre or mood:

```bash
python -m playlistgen --genre "Rap"
python -m playlistgen --mood "Epic"
python -m playlistgen --genre "Rap" --mood "Energetic"
```

Generate a mix from a seed song using Last.fm similarity:

```bash
python -m playlistgen seed-song --song "Miles Davis - Blue In Green" --num 20
```

Launch the interactive text UI:

```bash
python -m playlistgen gui
```

---

## 🔧 Configuration

Create a `config.yml` in the project root or `~/.playlistgen/config.yml` to override default settings (see the provided `config.yml` for details). Typical options include paths for your iTunes JSON, Spotify directory, and output folders.

Example:

```yaml
ITUNES_JSON: ./itunes_slimmed.json
SPOTIFY_DIR: ./spotify_history
PROFILE_PATH: ./taste_profile.json
OUTPUT_DIR: ./mixes
LASTFM_API_KEY: your_lastfm_api_key
CLUSTER_COUNT: 6
MAX_PER_ARTIST: 5
TRACKS_PER_MIX: 50
YEAR_MIX_ENABLED: true
YEAR_MIX_RANGE: 1
CACHE_DB: ~/.playlistgen/mood_cache.sqlite
SPOTIFY_REDIRECT_URI: http://localhost:8888/callback
```

The `SPOTIFY_REDIRECT_URI` value must match one of the redirect URIs
configured in your Spotify developer dashboard.

To use online genre/mood detection, set your **Last.fm API key** in the config.

---

## 🎼 How It Works

1. **iTunes Track Import:** Load and normalize your library’s track and genre metadata.
2. **Spotify Profile Build:** Analyze your play history for play/skip counts, artist/track/year preferences, and mood tags.
3. **Genre Recovery:** Fill missing genres using Last.fm tags.
4. **Track Scoring:** Apply custom scoring using plays, skips, artist and year affinity.
5. **Clustering:** TF-IDF clustering by artist, genre, and track name.
6. **Mix Creation:** Build year-based mixes (optional) and cluster-based mixes named with human-friendly mood and genre labels.
7. **M3U Export:** Save playlists into the `./mixes` folder.

---

## 🧠 Mood Detection (Patience is a Virtue)

* Fetches mood probabilities from Last.fm.
* Initial mood tagging can be very slow for large libraries.

---

### Train Playlist Clustering Model

Use the provided script to scrape playlists from Spotify and build a model for
scoring candidates later:

```bash
python -m playlistgen.train_model "rock" --limit 20 --output model.joblib
```

The script reads Spotify credentials from command line options or the
`SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` environment variables.

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
