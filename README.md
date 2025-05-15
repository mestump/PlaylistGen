# 🎧 PlaylistAgent — Smart Daily Mix Generator (Mostly AI Slop)

A command-line Python tool for generating smart, mood-aware, Daily-Mix style playlists using your **iTunes** and **Spotify** listening history. Inspired by Spotify's Daily Mix and iTunes metadata, this tool integrates your play/skip feedback, genres, and online mood classification to curate high-quality playlists with a personal touch.

---

## 🚀 Features

- 📦 **Dependency auto-check and install**
- 🔄 **iTunes XML → JSON converter**
- 🧠 **Spotify taste profile builder**
- 📊 **Daily scoring of tracks using play/skip feedback**
- 🌐 **Online genre lookup** (Last.fm → MusicBrainz)
- 🎨 **Genre normalization & clustering**
- 👥 **Artist diversity enforcement**
- 🧩 **Fallback back-fill logic to meet playlist length**
- 📂 **M3U playlist generation with intelligent naming**
- 💿 **Optional Year-based mix generation**

---

## 🧰 Requirements

- Python 3.7+
- pip packages: `numpy`, `pandas`, `scikit-learn`, `requests`, `hdbscan` 

The script will automatically install missing dependencies on the first run.


2. Run the script — it'll install any missing packages automatically.

---

## 📂 Usage

### Export your iTunes Library

1. Open iTunes (or Music app on macOS).
2. Go to `File` → `Library` → `Export Library`.
3. Save the exported XML file (e.g., `iTunes Music Library.xml`) somewhere convenient.
### Convert iTunes Library

Convert your iTunes `.xml` library into a JSON file:

```bash
python PlaylistAgent_mood_profile.py convert --input "iTunes Music Library.xml" --output itunes_slimmed.json
```

### Build Profile (Spotify)
Export your Spotify listening history from [Spotify Privacy](https://www.spotify.com/us/account/privacy/) and unzip to `./spotify_history`.

```bash
python PlaylistAgent_mood_profile.py --build-profile
```

### Generate Playlists

Create playlists by combining iTunes and Spotify data:

```bash
python PlaylistAgent_mood_profile.py
```

Optional flags:

* `--build-profile`: Rebuild Spotify profile
* `--force-refresh`: Rebuild profile even if cache exists

---

## 🔧 Configuration

Defaults are stored in the `CONFIG` dictionary in the script:

```python
CONFIG = {
  'ITUNES_JSON': './itunes_slimmed.json',
  'SPOTIFY_DIR': './spotify_history',
  'PROFILE_PATH': './taste_profile.json',
  'OUTPUT_DIR': './mixes',
  'LASTFM_API_KEY': 'your_lastfm_api_key',
  'CLUSTER_COUNT': 6,
  'MAX_PER_ARTIST': 5,
  'TRACKS_PER_MIX': 50,
  'YEAR_MIX_ENABLED': True,
  'YEAR_MIX_RANGE': 1,
}
```

To use online genre/mood detection, update your **Last.fm API key** in the config.

---

## 🎼 How It Works

1. **iTunes Track Import**: Load your library and normalize track/genre metadata.
2. **Spotify Profile Build**: Analyze your play history and extract:

   * Play/skip counts
   * Artist/track/year preferences
   * Mood tags via AcousticBrainz
3. **Genre Recovery**: Fill in missing genres using Last.fm tags.
4. **Track Scoring**: Based on a custom formula including:

   * Play counts
   * Skip penalties
   * Artist/year affinity
5. **Clustering**: TF-IDF clustering by artist, genre, and track name.
6. **Mix Creation**:

   * Year-based mix (optional)
   * Cluster-based mixes (named by top genres)
7. **M3U Export**: Playlists are saved to `./mixes` folder.

---

## 🧠 Mood Detection (Takes a LOOOOOOOONG time)

* **MusicBrainz** → Get MBIDs
* **AcousticBrainz** → Get mood probabilities
