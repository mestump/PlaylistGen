# PlaylistGen — AI-Powered Smart Playlist Generator

Generate Spotify Daily-Mix quality playlists from your **local music library**.
PlaylistGen scores and clusters your tracks using your listening history, local
audio analysis, and Claude AI — no streaming subscription required.

---

## What it does

PlaylistGen reads your iTunes / Music.app library (or a plain folder of music
files), analyses every track, and produces one or more themed M3U playlists.
Each run it:

1. Extracts embedded tags (BPM, genre, mood, year, play count, skip count)
   with **mutagen** — no internet required for this step.
2. Analyses audio features (tempo, energy, brightness) locally with
   **libROSA** — again, fully offline.
3. Optionally enriches mood/energy metadata via **Claude** (API or free
   paste-in workflow), or a **local Ollama model** running on your own
   machine — no API key or internet required for the Ollama path.
4. Builds a **session model** from your Spotify streaming-history JSON export,
   learning which tracks you play together and which you've played recently.
5. Scores every track by combining plays, skips, recency, and co-occurrence.
6. Clusters the top tracks by audio similarity, mood, or genre and writes an
   M3U file per cluster into `./mixes/`.
7. Optionally hands the scored library to **Claude** for fully themed
   playlist curation (e.g. "Late-Night Drive", "Morning Run").

---

## Quick start

### 1 — Install

```bash
git clone https://github.com/mestump/PlaylistGen.git
cd PlaylistGen
pip install -e .
```

> **Python 3.10+ is recommended.** All heavy dependencies (`librosa`, `numpy`,
> `scikit-learn`) are listed in `requirements.txt` and installed automatically.

### 2 — Export your iTunes library

1. Open **iTunes** or **Music.app**.
2. Go to **File → Library → Export Library**.
3. Save the `.xml` file in the project root (or anywhere — you'll be asked for
   the path on first run).

If you don't use iTunes, skip this step; PlaylistGen can scan a plain music
folder instead.

### 3 — (Optional) Export your Spotify listening history

Spotify's API no longer supports third-party playlist building, but you can
download your own data:

1. Go to [spotify.com → Account → Privacy settings](https://www.spotify.com/us/account/privacy/).
2. Request **Extended streaming history** and wait for the email (usually
   30 minutes to a few days).
3. Unzip the download. You'll get one or more files named
   `StreamingHistory_music_*.json` or `Streaming_History_Audio_*.json`.
4. Note the folder path — you'll enter it during setup.

The session model learns recency (how recently you played a track) and
co-occurrence (tracks you often play in the same session) without calling any
API.

### 4 — Run

```bash
python -m playlistgen gui        # interactive menu (recommended for first run)
python -m playlistgen            # non-interactive: generate a smart mix
```

On first run, the interactive menu detects that no library is configured and
launches a **setup wizard** that walks you through all the steps above.

### 5 — Find your playlists

Playlists are saved as `.m3u` files in `./mixes/` (configurable). Open them
with any media player that supports M3U, or sync to an iPod / DAP.

---

## Using the interactive menu

Launch with:

```bash
python -m playlistgen gui
```

The menu is divided into four sections:

### Generate Playlists

| Option | What it does |
|--------|-------------|
| **Smart mix** | Runs the full pipeline; auto-selects the best clustering strategy (audio features → mood → genre) |
| **Mix from a seed song** | Finds tracks that are acoustically similar to one song you name |
| **Filter by mood** | Generates a playlist limited to a single mood (Happy, Chill, Energetic, etc.) |
| **Filter by genre** | Generates a playlist limited to a single genre |

### AI Features *(Anthropic API key required)*

| Option | What it does |
|--------|-------------|
| **Claude: Smart playlist curation** | Sends your top-scored tracks to Claude (Sonnet) which groups them into 5–10 themed playlists with descriptive names |
| **Claude: Enrich library metadata** | Sends tracks in batches of 100 to Claude (Haiku) which classifies each track's Mood, Energy, and Valence. Results are cached in SQLite so only new tracks are re-analysed |

### AI Features *(local Ollama — fully offline, no API key)*

| Option | What it does |
|--------|-------------|
| **Ollama: Smart playlist curation** | Same curation workflow as Claude but using a local Ollama model. Requires `OLLAMA_BASE_URL` to be set |
| **Ollama: Enrich library metadata** | Batch mood/energy enrichment via local model — used automatically as a fallback when Claude is unavailable |

### AI Features *(no API key needed)*

| Option | What it does |
|--------|-------------|
| **Generate Claude session file** | Writes a single `.md` file containing **all** pending enrichment batches. Upload it to Claude.ai and Claude processes them sequentially, saving a JSON artifact per batch — ideal for large libraries |
| **Generate AI enrichment prompt** | Writes a single-batch prompt you paste into ChatGPT, Gemini, etc. Run again after each import to get the next batch |
| **Generate AI curation prompt** | Same paste-in workflow for full playlist curation — the AI groups your tracks into themed playlists, you paste the response back, and the M3Us are written immediately |
| **Import AI response from file** | Import a previously generated prompt file or raw JSON artifact — auto-detects enrich vs curate mode |

### Spotify

| Option | What it does |
|--------|-------------|
| **Export playlist to Spotify** | Pick any generated M3U and push it directly to your Spotify account as a new playlist. Requires `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` in config |

### Setup & Maintenance

| Option | What it does |
|--------|-------------|
| **Configure API keys & data paths** | Set your Anthropic key, Last.fm key, Ollama URL, Spotify credentials, etc. |
| **Re-configure Spotify history** | Update the Spotify streaming history path with live validation — shows play count and date range immediately, lets you retry if the path is wrong |
| **Refresh metadata cache** | Re-fetches Last.fm tags for tracks not yet in the cache (only new tracks are fetched) |

### Advanced

| Option | What it does |
|--------|-------------|
| **Edit a config value** | Change any config key directly (shows categorised list) |

---

## CLI reference

```
python -m playlistgen [subcommand] [options]
```

Running without a subcommand generates a smart mix using your current config.

### Global options

| Flag | Default | Description |
|------|---------|-------------|
| `--log-level LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `--genre GENRE` | — | Restrict output to tracks matching this genre |
| `--mood MOOD` | — | Restrict output to tracks matching this mood |
| `--library-dir PATH` | — | Scan a folder of music files instead of iTunes |
| `--ai-enrich` | off | Run Claude batch enrichment before clustering |
| `--ai-curate` | off | Use Claude for full playlist curation instead of algorithmic clustering |
| `--no-lastfm` | off | Disable Last.fm tag fetching entirely |

### Subcommands

```bash
# Launch the interactive menu
python -m playlistgen gui

# Build a playlist around one track
python -m playlistgen seed-song --song "Artist - Title" --num 20

# Force-refresh the metadata cache (Last.fm tags)
python -m playlistgen recache-moods

# Discover music similar to a genre query (requires Spotify credentials)
python -m playlistgen discover --genre "Indie Rock" --limit 5

# Generate a Claude session file — all batches in one upload (recommended for Claude)
python -m playlistgen export-ai-session
python -m playlistgen export-ai-session --batch-size 500   # larger batches if your library is huge

# Generate a single-batch enrichment prompt (for ChatGPT / Gemini / other AIs)
python -m playlistgen export-ai-prompt --mode enrich
python -m playlistgen export-ai-prompt --mode enrich --batch-size 100   # smaller for Gemini free

# Generate a paste-in AI curation prompt
python -m playlistgen export-ai-prompt --mode curate --n-playlists 8

# Import any AI response — plain JSON artifact from Claude.ai, or a .txt prompt file
python -m playlistgen import-ai-result batch_1_enrichment.json
python -m playlistgen import-ai-result playlistgen_enrich_prompt.txt

# Export a generated playlist to Spotify (requires Spotify app credentials in config)
python -m playlistgen spotify-export ./mixes/Late-Night\ Drive.m3u
python -m playlistgen spotify-export ./mixes/Morning\ Run.m3u --name "My Morning Run" --private
```

### Examples

```bash
# Quick smart mix
python -m playlistgen

# Chill playlist
python -m playlistgen --mood "Chill"

# Rock genre, verbose logging
python -m playlistgen --genre "Rock" --log-level DEBUG

# AI-curated playlists from top tracks
python -m playlistgen --ai-curate

# Enrich library metadata then generate
python -m playlistgen --ai-enrich

# Seed playlist: 30 tracks similar to "Blue In Green"
python -m playlistgen seed-song --song "Miles Davis - Blue In Green" --num 30

# Scan a music folder directly (no iTunes needed)
python -m playlistgen --library-dir ~/Music/FLAC
```

---

## Configuration

PlaylistGen looks for a config file in this order:

1. `./config.yml` (project root)
2. `~/.playlistgen/config.yml`

All keys can also be set as **environment variables** (same name, upper case).
The interactive menu writes to whichever config file was found (or creates
`~/.playlistgen/config.yml` if neither exists).

### Full config reference

#### Library

| Key | Default | Description |
|-----|---------|-------------|
| `ITUNES_JSON` | `./itunes_slimmed.json` | Path to cached iTunes JSON (auto-generated from XML) |
| `ITUNES_XML` | *(none)* | Path to raw iTunes Library XML export |
| `LIBRARY_DIR` | *(none)* | Scan this folder instead of iTunes |
| `MUTAGEN_ENABLED` | `true` | Extract BPM, mood, genre tags from embedded audio tags |

#### Output

| Key | Default | Description |
|-----|---------|-------------|
| `OUTPUT_DIR` | `./mixes` | Where to write M3U playlist files |
| `TRACKS_PER_MIX` | `50` | Target number of tracks per playlist |
| `MAX_PER_ARTIST` | `4` | Max tracks per artist per playlist (diversity) |

#### Clustering

| Key | Default | Description |
|-----|---------|-------------|
| `CLUSTER_COUNT` | `6` | Number of clusters / playlists to generate |
| `CLUSTER_STRATEGY` | `auto` | `auto`, `audio`, `mood`, or `tfidf`. Auto selects based on data coverage |
| `CLUSTER_HYBRID` | `false` | Mood grouping first, then audio sub-clusters within each mood |
| `YEAR_MIX_ENABLED` | `true` | Also generate a year-based mix |
| `YEAR_MIX_RANGE` | `1` | ±years around the dominant year for year-based mix |

#### AI Features (Anthropic)

| Key | Default | Description |
|-----|---------|-------------|
| `ANTHROPIC_API_KEY` | *(none)* | Your Anthropic API key (`sk-ant-...`) |
| `AI_ENHANCE` | `false` | Enable AI playlist naming |
| `AI_BATCH_ENRICH` | `false` | Run Claude batch mood enrichment in the pipeline |
| `AI_CURATE` | `false` | Use Claude for full playlist curation |
| `AI_MODEL` | `claude-haiku-4-5-20251001` | Model for batch enrichment (cheap, fast) |
| `AI_CURATE_MODEL` | `claude-sonnet-4-6` | Model for full curation (more capable) |
| `AI_ENRICH_CACHE_DB` | `~/.playlistgen/claude_enrichment.sqlite` | SQLite cache for enrichment results |

#### AI Features (local Ollama)

| Key | Default | Description |
|-----|---------|-------------|
| `OLLAMA_BASE_URL` | *(none)* | Ollama server URL (e.g. `http://localhost:11434`). Enables Ollama backend when set |
| `OLLAMA_MODEL` | `hf.co/unsloth/Qwen3.5-35B-A3B-GGUF:UD-IQ2_XXS` | Default Ollama model |
| `OLLAMA_ENRICH_MODEL` | *(same as OLLAMA_MODEL)* | Model used for batch metadata enrichment |
| `OLLAMA_CURATE_MODEL` | *(same as OLLAMA_MODEL)* | Model used for playlist curation |

#### Spotify Export

| Key | Default | Description |
|-----|---------|-------------|
| `SPOTIFY_CLIENT_ID` | *(none)* | Spotify app client ID (create at [developer.spotify.com](https://developer.spotify.com/dashboard)) |
| `SPOTIFY_CLIENT_SECRET` | *(none)* | Spotify app client secret |
| `SPOTIFY_REDIRECT_URI` | `http://localhost:8888/callback` | OAuth redirect URI — must match your Spotify app settings |

#### Session Model (Spotify history)

| Key | Default | Description |
|-----|---------|-------------|
| `SPOTIFY_HISTORY_PATH` | *(none)* | Path to `StreamingHistory*.json` file or folder |
| `SESSION_GAP_MINUTES` | `30` | Listening gap (minutes) that splits one session from the next |
| `RECENCY_HALF_LIFE_DAYS` | `90` | How quickly recency scores decay (90 days → half weight) |

#### Local Audio Analysis (libROSA)

| Key | Default | Description |
|-----|---------|-------------|
| `LIBROSA_ENABLED` | `true` | Run local audio feature extraction (BPM, energy, brightness) |
| `AUDIO_CACHE_DB` | `~/.playlistgen/audio.sqlite` | SQLite cache for audio analysis results |
| `AUDIO_ANALYSIS_WORKERS` | `4` | Parallel threads for audio analysis |

#### Last.fm (optional fallback)

| Key | Default | Description |
|-----|---------|-------------|
| `LASTFM_API_KEY` | *(none)* | Last.fm API key (register at last.fm/api/account/create) |
| `LASTFM_CACHE_DB` | `~/.playlistgen/lastfm.sqlite` | SQLite cache for Last.fm tag results |
| `LASTFM_RATE_LIMIT_MS` | `200` | Milliseconds between Last.fm API calls |
| `MOOD_CONCURRENCY` | `10` | Concurrent Last.fm requests |

#### Taste profile & feedback

| Key | Default | Description |
|-----|---------|-------------|
| `PROFILE_PATH` | `./taste_profile.json` | Legacy Spotify-derived taste profile (if present) |
| `FEEDBACK_PATH` | `~/.playlistgen/feedback.json` | Stores skip/like signals from previous runs |

---

## How it works

```
iTunes XML / music folder
        │
        ▼
  1. Library load        mutagen extracts embedded tags (BPM, genre, year,
                         play count, skip count, mood, rating)
        │
        ▼
  2. Audio analysis      libROSA analyses each audio file locally:
                         BPM · RMS energy · spectral brightness · zero-crossing rate
                         Results cached in SQLite (only new/changed files re-analysed)
        │
        ▼
  3. Metadata enrichment (priority order)
         a. Claude batch enrichment (if AI_BATCH_ENRICH=true)
            → sends 100 tracks at a time to Claude Haiku
            → classifies Mood, Energy (1–10), Valence (1–10)
            → cached indefinitely
         b. Ollama batch enrichment (if OLLAMA_BASE_URL set, used as fallback)
            → same classification via local model, fully offline
         c. Last.fm tag lookup (if LASTFM_API_KEY set and mood still unknown)
         d. Embedded tag fallback (mutagen Mood/Comment field)
        │
        ▼
  4. Session model       Reads Spotify StreamingHistory JSON files
                         → splits plays into listening sessions (30-min gap)
                         → builds co-occurrence matrix (tracks played together)
                         → calculates recency scores (exponential decay)
        │
        ▼
  5. Scoring             Each track gets a composite score:
                         base = log(plays+1) − skip_penalty + genre_affinity
                         × recency_multiplier (1.0–1.5×)
                         + co_occurrence_boost (tracks played with top favourites)
        │
        ▼
  6. Clustering          Strategy selected automatically or by config:
                         audio   → KMeans on [BPM, energy, brightness, ZCR]
                         mood    → group by mood label, then sub-cluster
                         hybrid  → mood first, audio sub-clusters within mood
                         tfidf   → TF-IDF on artist + genre + track name
        │
        ▼
  7. Playlist build      • Top N tracks per cluster
                         • Artist diversity enforced (MAX_PER_ARTIST)
                         • Back-fill to meet TRACKS_PER_MIX
                         • Claude names each playlist (if AI_ENHANCE=true)
        │
        ▼
  8. M3U export          One .m3u file per playlist → OUTPUT_DIR/
        │
        ▼
  9. Spotify export (optional)
                         Push any M3U playlist directly to your Spotify account
                         via spotipy OAuth (SPOTIFY_CLIENT_ID / SECRET required)
```

---

## AI features in depth

### Claude: Batch library enrichment

Sends your track list to Claude in batches of 100. Each batch gets a prompt
asking for Mood, Energy (1–10), and Valence (1–10) for every track. Results
are stored in a local SQLite database — subsequent runs skip already-enriched
tracks.

**Cost estimate:** ~$0.001 per track using Claude Haiku.
**Privacy:** Only artist name, title, album, and year are sent — no audio
data leaves your machine.

Enable via config (`AI_BATCH_ENRICH: true`) or CLI (`--ai-enrich`) or the
interactive menu.

### Claude: Full playlist curation

Sends the top 300 scored tracks to Claude Sonnet with a prompt asking it to
group them into themed playlists. Claude returns JSON describing each playlist
(name, description, track list). This replaces the algorithmic clustering step
entirely when enabled.

Enable via config (`AI_CURATE: true`) or CLI (`--ai-curate`) or the
interactive menu.

### Ollama: local AI (fully offline)

If you have [Ollama](https://ollama.com) running locally, set `OLLAMA_BASE_URL`
in your config and PlaylistGen will use it as a drop-in replacement for Claude
— no API key, no internet, no cost per track.

```yaml
OLLAMA_BASE_URL: http://localhost:11434
OLLAMA_ENRICH_MODEL: hf.co/unsloth/Qwen3.5-35B-A3B-GGUF:UD-IQ2_XXS
```

Ollama is used **automatically as a fallback** in the pipeline when Claude
enrichment is unavailable. You can also trigger it explicitly from the GUI
menu or set it as the primary enricher.

### Spotify export

Push any generated M3U playlist directly to your Spotify library:

```bash
python -m playlistgen spotify-export ./mixes/Late-Night\ Drive.m3u
python -m playlistgen spotify-export ./mixes/Morning\ Run.m3u --name "My Morning Run" --private
```

Or use the **Export playlist to Spotify** option in the GUI menu.

**Setup** — create a Spotify app at [developer.spotify.com](https://developer.spotify.com/dashboard),
add `http://localhost:8888/callback` as a redirect URI, then set in config:

```yaml
SPOTIFY_CLIENT_ID: your_client_id
SPOTIFY_CLIENT_SECRET: your_client_secret
```

The first export opens a browser for OAuth authorisation. The token is cached
for subsequent runs.

### AI features without an API key (paste-in workflow)

If you don't have an Anthropic API key — or prefer to use your existing
Claude.ai, ChatGPT Plus, or Gemini Advanced subscription — PlaylistGen can
generate prompt files you paste into any AI and import the response back.

#### Claude.ai — session file (recommended for large libraries)

The session file puts **all pending batches in a single `.md` file** you
upload to Claude.ai. Claude processes each batch sequentially and saves a
downloadable JSON artifact per batch — no copy-pasting required.

```bash
# 1. Generate the session file (all batches, auto-sized)
python -m playlistgen export-ai-session

# → Writes playlistgen_enrichment_session.md

# 2. Upload to Claude.ai (drag & drop the .md file into the chat)
# 3. Send: "Please process this enrichment session"
# 4. Claude works through each batch and creates batch_N_enrichment.json artifacts
# 5. Download each artifact as Claude creates it, then import:

python -m playlistgen import-ai-result batch_1_enrichment.json
python -m playlistgen import-ai-result batch_2_enrichment.json
# ... repeat for each batch

# 6. Run normally — enriched data is used automatically.
python -m playlistgen
```

Each import is independent and idempotent — you can import batch 2 while
Claude is still working on batch 3, and re-running import on an already-
imported file is a safe no-op.

#### Other AIs (ChatGPT, Gemini) — single-batch paste-in

For AIs that don't support file artifacts, generate one batch at a time,
paste the prompt, and paste the response back into the file:

```bash
# Generate one batch
python -m playlistgen export-ai-prompt --mode enrich

# → playlistgen_enrich_prompt.txt
# Copy the PROMPT START/END section → paste into AI → copy JSON response
# Paste response into the RESPONSE section at the bottom of the file, then:

python -m playlistgen import-ai-result playlistgen_enrich_prompt.txt

# Run again to get the next batch (already-imported tracks are skipped)
python -m playlistgen export-ai-prompt --mode enrich
# → repeat until all tracks are enriched
```

**Curation workflow (any AI):**

```bash
python -m playlistgen export-ai-prompt --mode curate --n-playlists 6
# → paste into AI, paste response back into file
python -m playlistgen import-ai-result playlistgen_curate_prompt.txt
# → M3U playlists written to ./mixes/ immediately
```

**Recommended batch sizes by AI:**

| AI | Output token limit | Tracks per batch | Notes |
|----|-------------------|-----------------|-------|
| Claude.ai (any plan) | 32 K | up to **500** | Use session file — no copy-pasting |
| ChatGPT Plus (GPT-4o) | 16 K | up to **250** | `--batch-size 250` |
| Gemini Advanced | 8 K | up to **100** | `--batch-size 100` |

The import is idempotent — re-importing an already-cached track is a no-op.

---

## Session model in depth

The session model replaces the old (now-defunct) Spotify API discovery. It
reads the JSON files from your personal Spotify data export and learns two
things:

**Recency scores** — tracks you've played recently are scored higher. The
score decays with a 90-day half-life by default:

```
recency_score = exp(−ln(2) × days_since_last_play / half_life)
```

**Co-occurrence scores** — if track A and track B are often played in the same
listening session, they are considered related. Tracks that co-occur with your
top 50 most-played tracks get a scoring boost.

Both signals feed into the track scorer as multipliers / additive bonuses, so
they influence which tracks make it into each playlist without overriding the
primary play/skip quality signal.

---

## Audio analysis in depth

libROSA analyses the first 2 minutes of each audio file to extract:

| Feature | What it captures |
|---------|-----------------|
| BPM | Tempo — how fast the track feels |
| Energy (RMS) | Loudness / intensity |
| Spectral brightness | Treble-heaviness — bright vs. warm/dark |
| Zero-crossing rate | Noisiness / distortion — useful for genre separation |

These four features are normalised to [0, 1] and fed into KMeans clustering,
producing groups of tracks that *sound* similar rather than just sharing a
genre tag.

Results are cached in SQLite keyed by file path + modification time, so only
new or changed files are re-analysed.

---

## Development

### Running tests

```bash
python -m pytest tests/ -v
```

425 tests across unit and integration suites. All pass on Python 3.10+.

### Project structure

```
PlaylistGen/
├── playlistgen/
│   ├── __init__.py          public API exports
│   ├── __main__.py          entry point (python -m playlistgen)
│   ├── cli.py               argparse CLI
│   ├── gui.py               questionary interactive menu
│   ├── pipeline.py          10-stage orchestration pipeline
│   ├── config.py            config loading / saving
│   ├── utils.py             path/URL validation and shared helpers
│   ├── itunes.py            iTunes XML → DataFrame
│   ├── metadata.py          tag extraction and enrichment helpers
│   ├── audio_analysis.py    libROSA feature extraction + SQLite cache (ProcessPoolExecutor)
│   ├── session_model.py     Spotify history → co-occurrence + recency
│   ├── llm_client.py        dispatcher: routes AI calls to Claude or Ollama
│   ├── ai_enhancer.py       Claude batch enrichment + curation (API path)
│   ├── enrichers/
│   │   └── ollama_enricher.py  Ollama batch metadata enrichment (fully offline)
│   ├── prompt_io.py         Paste-in AI workflow — prompt export + response import
│   ├── clustering.py        KMeans / mood / TF-IDF clustering
│   ├── scoring.py           composite track scorer (vectorized)
│   ├── playlist_builder.py  M3U writer
│   ├── playlist_scraper.py  Spotify track discovery via API
│   ├── seed_playlist.py     seed-song similarity search
│   ├── similarity.py        audio feature similarity helpers
│   ├── pattern_analyzer.py  listening pattern analysis
│   ├── spotify_export.py    push M3U playlists to Spotify via OAuth
│   ├── spotify_profile.py   Spotify taste profile helpers
│   ├── tag_mood_service.py  Last.fm tag cache
│   └── train_model.py       (legacy) Spotify scraping model trainer
└── tests/
    ├── test_audio_analysis.py
    ├── test_session_model.py
    ├── test_clustering*.py
    ├── test_scoring*.py
    └── ...  (425 tests total)
```

---

## Troubleshooting

**"No tracks found"** — Check that `ITUNES_JSON` or `LIBRARY_DIR` points to
the right location. Run with `--log-level DEBUG` to see exactly what paths are
being scanned.

**Audio analysis is slow** — The first run analyses every file. Subsequent
runs are instant (cached). Increase `AUDIO_ANALYSIS_WORKERS` if you have more
CPU cores available.

**Don't have an Anthropic API key?** — Use the paste-in workflow. For
Claude.ai, run `python -m playlistgen export-ai-session` to generate a single
file covering your whole library, upload it, and import each batch artifact.
For ChatGPT/Gemini, run `export-ai-prompt --mode enrich` and paste one batch
at a time.

**Spotify history didn't import correctly at setup** — Go to
`Setup → Re-configure Spotify history` in the interactive menu. It validates
the path immediately and shows the play count and date range so you can
confirm it loaded before saving.

**Claude enrichment is expensive** — Use Claude Haiku (`claude-haiku-4-5-20251001`)
which is the default for enrichment. Sonnet is only used for full curation.
Costs are roughly $0.001 per track for enrichment (one-time, cached).
Alternatively, use the free paste-in workflow described above.

**Last.fm rate limits** — Increase `LASTFM_RATE_LIMIT_MS` (default 200 ms).
Last.fm allows ~5 requests/second on a free key.

**Playlists are too similar** — Reduce `MAX_PER_ARTIST` or increase
`CLUSTER_COUNT`. Try `CLUSTER_STRATEGY: audio` for more acoustically distinct
groups.

---

## License

Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
See [LICENSE](LICENSE) for details.
