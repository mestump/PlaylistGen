# pgen — local listening-history playlist generator

`pgen` builds Daily-Mix-style playlists from **your** music library and **your**
listening history, entirely on your machine. No accounts, no cloud services, no
API keys. Playlists are seeded and reproducible, and every part of the database
except the raw play events can be rebuilt from scratch.

```sh
pip install git+https://github.com/mestump/PlaylistGen.git
pgen ingest files --library ~/Music   # index your library
pgen build                            # materialize play signals + affinity
pgen make                             # write ~/.local/share/pgen/playlists/daily-*.m3u
```

## Requirements

- Python **3.11 or newer** (3.11–3.15 tested)
- Linux or macOS. Windows works in principle (pure-Python, `pathlib` throughout)
  but is untested — reports welcome.
- Runtime dependencies: `mutagen` (tag reading) and `duckdb` (optional parquet
  enrichment). Both install automatically with the package. Nothing else; there
  is no ffmpeg/librosa/numpy requirement.

## Install

Pick one:

```sh
# from the repository head
pip install git+https://github.com/mestump/PlaylistGen.git

# pinned, from the release assets (wheel or sdist)
pip install https://github.com/mestump/PlaylistGen/releases/download/v2.0.0/playlistgen_v2-2.0.0-py3-none-any.whl

# from a clone (editable, with the test suite)
git clone https://github.com/mestump/PlaylistGen && cd PlaylistGen
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest            # 12 tests, no network, no fixtures needed
```

This installs one console script, `pgen`. Upgrade with `--upgrade` /
`--force-reinstall`; uninstall with `pip uninstall playlistgen-v2`. Your data
(`~/.local/share/pgen/`) is never touched by install/uninstall.

## Quickstart

```sh
pgen ingest files --library /path/to/your/music
pgen build
pgen make
```

That is the entire core loop. The playlist appears in
`~/.local/share/pgen/playlists/` as an `.m3u` with absolute paths — open it in
any player. With no listening history ingested yet, `make` still works: it
produces a diversity-capped, coherence-ranked sampler of your library (see
[Cold start](#cold-start-no-listening-history)).

Add history sources over time; each one sharpens the result. Re-running any
ingest is safe — sources are stamped and skipped when unchanged.

## Concepts

- **Store** — one SQLite database (WAL mode) at `~/.local/share/pgen/v2.db`
  (override per-command with `--db PATH`). Append-only `play_events` +
  per-source `source_counts`; tracks are identified by normalized
  `(artist, title)` so the same song ingested from three sources is one row.
- **Projections** — `track_signal`, `track_affinity`, `track_genre`, `features`
  are derived views, rebuilt by `pgen build` / `pgen genres` / `pgen features`.
  Delete the store and re-ingest to start over; nothing outside the store and
  the playlist folder is ever written.
- **Idempotency** — every source records a stamp (file mtime); re-ingesting an
  unchanged source is a no-op. `--force` re-reads anyway.
- **Read-only neighbors** — player databases (Strawberry) are opened `ATTACH`
  read-only-style and never written.

## Command reference

Global flag, valid before any subcommand:

| flag | default | meaning |
|---|---|---|
| `--db PATH` | `~/.local/share/pgen/v2.db` | store location (created on first use) |

### `pgen ingest SOURCE [SOURCE ...] [options]

Sources: `files`, `strawberry`, `ipod`, `spotify-history`, or `all` (expands to
the four). Sources run in the order given; each prints a JSON summary, and a
final `totals:` line shows cumulative store counts.

| option | default | applies to | meaning |
|---|---|---|---|
| `--library DIR` | `~/Music` | `files` | directory tree to index recursively |
| `--strawberry-db PATH` | `~/.local/share/strawberry/strawberry/strawberry.db` | `strawberry`, `ipod` | Strawberry database; opened read-only |
| `--ipod-sb-db PATH` | (same as `--strawberry-db`) | `ipod` | alternate db holding the `device_1_songs` table |
| `--spotify-glob GLOB [GLOB ...]` | *(required for the source)* | `spotify-history` | one or more paths/globs of `Streaming_History_Audio_*.json` |
| `--force` | off | all | re-ingest even if the source stamp is unchanged |

Per-source notes:

- **files** — indexes audio files (`.mp3 .m4a .aac .flac .ogg .opus .wma .wav
  .alac`), artist/title/album/year from tags via mutagen; files without tags
  fall back to `Artist - Title` filename parsing. ALAC-in-m4a and MP3 are
  ordinary supported inputs. Indexing is metadata-only; no audio is decoded.
- **strawberry** — imports play/skip counters from Strawberry's `songs` table
  into `source_counts`. If the DB is locked by a running player, retry later;
  nothing is corrupted by a failed attempt.
- **ipod** — play/skip counts from the `device_1_songs` table Strawberry wrote
  at the device's last sync, joined by normalized identity. Requires a prior
  Strawberry→iPod sync and the Strawberry DB to be reachable; point
  `--ipod-sb-db` at a copy if the live DB is busy.
- **spotify-history** — Spotify **Extended Streaming History** JSON
  (`Streaming_History_Audio_YYYY-YYYY_N.json` from the privacy download or
  the newer per-year files). Timestamped plays land in `play_events` — the
  strongest signal the scorer uses. Pass the glob quoted so your shell doesn't
  expand it: `--spotify-glob '~/sync/spot/Streaming_History_Audio_*.json'`.

### `pgen status`

Prints store totals: track count, per-source event counts with the
first→last timestamp span, and per-source counter totals. Use it to confirm a
source actually landed.

### `pgen build [--skip-affinity]`

Materializes the projections `make` consumes:

- **signals** (`track_signal`) — per-track plays over 90d/365d windows,
  total plays (timestamped events preferred, counter sources only where no
  events exist), skip ratio, days since last play, artist rank, spoken-word
  flag.
- **affinity** (`track_affinity`, unless `--skip-affinity`) — listening
  sessions cut at a 30-minute gap, then track–track co-occurrence PMI
  (minimum 2 co-plays, top-200 neighbors per track).

Run after every ingest that added events; it is a full rebuild (seconds on
six-digit event counts) and idempotent.

### `pgen make [-n N] [--seed S] [--scene FILTER]`

Generates one playlist and writes it to disk.

| flag | default | meaning |
|---|---|---|
| `-n INT` | `30` | target number of `#EXTINF` entries |
| `--seed INT` | `0` | RNG seed; same seed + same store = byte-identical playlist |
| `--scene TEXT` | *(none)* | genre filter — substring match (`jazz`), or comma-separated exact multi-genre match (`jazz,fusion`) |

How it selects: seeds = your top-weighted tracks (plays, recency, skip
penalty); candidates expand across affinity hops with exponential decay; each
candidate is scored on affinity + popularity + feature centricity, penalized
for skips, capped per artist, ordered by a coherence chain (genre overlap +
feature-vector cosine + affinity) with limited section breaks; familiar/novel
split ≈ 60/40.

Output: the `.m3u` path is printed inside a JSON report (`picked`,
`candidates`, artist histogram, coherence stats) and logged to the `runs`
table. Filename pattern:
`daily-YYYYmmdd-HHMMSS-s<seed>[-<scene>].m3u`.

Exit codes: `0` playlist written; `1` nothing selectable (empty library, or
`--scene` matched no genres — check `pgen status` / re-ingest); `2` usage
errors.

### `pgen genres --parquet-dir DIR`

Optional. Imports per-track artist-genre tags from a Spotify-style parquet
dump (`tracks.parquet`, `track_artists.parquet`, `artist_genres.parquet`) by
matching normalized `(artist, title)` — accents stripped, punctuation removed,
parentheticals dropped. Writes `track_genre`; `make --scene` and the coherence
term light up after this. Rebuilding replaces the table wholesale.

### `pgen features --parquet-dir DIR [--limit N] [--rebuild] [--threads 8] [--mem-limit 3GB]`

Optional. Same match, then scalar audio features per track
(`tempo energy danceability valence loudness speechiness acousticness
instrumentalness liveness key mode time_signature popularity`) from
`track_audio_features.parquet`, written into `features`. The expensive
identity scan is cached to `~/pgen-tmp/match-cache.parquet` — reruns only
process newly appeared keys. Streamed with DuckDB; sized for hundreds of GB of
parquet at a few GB of RAM.

Both commands accept the directory via `--parquet-dir` or the `PGEN_PARQUET_DIR`
environment variable. The public `spotify-data` dataset dumps match the
expected schema. **Neither is required** — `make` uses whatever exists.

## Data layout

| path | contents |
|---|---|
| `~/.local/share/pgen/v2.db` | the store (SQLite, WAL; `+wal`/`+shm` siblings) |
| `~/.local/share/pgen/playlists/` | generated `.m3u` files |
| `~/pgen-tmp/` | parquet-match caches (safe to delete; rebuilt on demand) |
| `$PGEN_PARQUET_DIR` | optional default for `genres`/`features` |

## Recipes

```sh
# everything a music hoarder can throw at it, then the daily mix
pgen ingest all --library /big/music \
  --spotify-glob '~/exports/Streaming_History_Audio_*.json'
pgen build && pgen make

# three fixed scene channels, reproducible
pgen make --scene 'post-rock' --seed 11 -n 40
pgen make --scene 'ambient,modern classical' --seed 12 -n 40

# nightly cron (0x0d * * *), store and playlists in default locations
pgen ingest spotify-history --spotify-glob '~/sync/Streaming_History_Audio_*.json' && \
pgen build && pgen make

# start over (settings live in env/args; there is nothing else)
rm -rf ~/.local/share/pgen
```

## Cold start (no listening history)

`make` on a store with zero play events still returns a playlist: every
indexed, non-spoken track is a candidate, the artist cap and coherence
ordering still apply, and ranking falls back to genre/feature coherence. The
result is a sampler, not a recommendation engine — ingest history for the real
thing. The `familiar` count in the report is `0` in this mode.

## Troubleshooting

- **`no such table: track_signal`** — run `pgen build` (any version before
  2.0.0 crashed here; 2.0.0 creates empty projections on demand).
- **`picked: 0`, exit 1** — no tracks with paths (`ingest files` first), or
  `--scene` matched no genre (only possible after `pgen genres`; drop the
  flag or broaden it).
- **`strawberry db not found`** — pass `--strawberry-db`; default assumes the
  standard `~/.local/share/strawberry/strawberry/strawberry.db` layout.
- **`no history files match`** — quote the glob and check the path; the
  source is skipped, not fatal, if you remove it from the invocation.
- **Network mounts** — index local disks or fast mounts; a cold NAS mount can
  stall the first `ingest files` walk (autofs behaves this way).
- **Duplicate-ish entries** — identity is normalized `(artist, title)`;
  differing artist spellings remain distinct by design. Fix tags at the
  source and re-ingest with `--force`.

## Version history

- **v2** (this): ground-up rewrite — event store, seeded affinity ranking, no
  accounts. Ships as the `playlistgen-v2` package / `pgen` command.
- **v1**: iTunes-XML pipeline with optional Last.fm/AI enrichment; retired.
  Preserved on the
  [`v1-legacy`](https://github.com/mestump/PlaylistGen/tree/v1-legacy) branch
  (tag `v1-archive`). Its packaging (`pip install -e .`) and audio-analysis
  paths were broken; do not build new work on it.

## License

[MIT](LICENSE)
