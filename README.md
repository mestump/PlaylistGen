# pgen — local listening-history playlist generator

`pgen` builds Daily-Mix-style playlists from **your** music library and **your**
listening history, entirely on your machine. No accounts, no cloud, no API keys.
Playlists are reproducible (seeded) and everything except the raw play events
can be rebuilt from scratch.

```sh
pip install git+https://github.com/mestump/PlaylistGen.git
pgen ingest files --library ~/Music   # index your library (tags via mutagen)
pgen build                            # materialize play signals + co-occurrence affinity
pgen make                             # write ~/.local/share/pgen/playlists/daily-*.m3u
```

Requires Python 3.11+. Runtime deps: `mutagen` and `duckdb` (installed
automatically). `pgen --help` lists every subcommand.

## How it works

1. **Ingest** — index music files, optionally plus plays/skips from Strawberry's
   database, an iPod's `iTunesDB`/`Play Counts` files, and Spotify "Extended
   Streaming History" JSON exports. Sources are idempotent (stamped by mtime);
   player databases are opened read-only, never written.
2. **Build** — per-track play signals (recency windows, skip ratio, artist rank)
   plus track–track co-occurrence affinity from listening sessions (30-minute
   gap), PMI-weighted.
3. **Make** — seeded affinity walk from your top tracks, blended with feature
   centricity, an artist diversity cap, a skip penalty, and a coherence-chained
   final order. Output is an `.m3u` with absolute paths plus a quality report.

## Ingest sources

| command | source | notes |
|---|---|---|
| `pgen ingest files --library DIR` | music files | mp3/m4a/aac/flac/ogg/opus/wma/wav; falls back to `Artist - Title` filename parsing |
| `pgen ingest strawberry` | Strawberry player DB | play/skip counters; default `~/.local/share/strawberry/strawberry/strawberry.db` |
| `pgen ingest ipod` | iPod `iTunesDB` + `Play Counts` | autodetects mounted iPods under `/run/media`; device files are authoritative — never trust player-app device caches |
| `pgen ingest spotify-history --spotify-glob '.../Streaming_History_Audio_*.json'` | Spotify Extended Streaming History | timestamped plays, the strongest signal |
| `pgen status` | — | per-source event/count totals |

Any one source is enough; add more whenever. Re-ingest is safe and cheap.
Without `--library`, `pgen ingest files` scans `~/Music`.

## Optional: audio features and genres from a Spotify parquet dump

If you have a parquet copy of the Spotify catalog (tables `tracks`,
`track_artists`, `artist_genres`, `track_audio_features` — the public
`spotify-data` dumps match), pgen can attach scalar audio features
(tempo/energy/valence/...) and artist genres by matching normalized
`(artist, title)`:

```sh
pgen genres   --parquet-dir /path/to/parquet
pgen features --parquet-dir /path/to/parquet
```

Set `PGEN_PARQUET_DIR` to make it sticky. Both steps are optional: `pgen make`
uses whatever exists and degrades to genre/feature-less ranking otherwise. A
fresh library with no listening history at all still produces a playlist
(cold start: diversity-capped, coherence-ranked pick from pathed tracks).

## Day-to-day

```sh
pgen make --seed 42              # same playlist every time
pgen make -n 50                  # longer
pgen make --scene jazz           # genre-substring filter
pgen make --scene 'jazz,fusion'  # exact multi-genre match
```

Playlists land in `~/.local/share/pgen/playlists/`; the filename carries the
seed and scene. The store is a single SQLite database (WAL) at
`~/.local/share/pgen/v2.db` — delete it and re-ingest to start over. Nothing
outside those two paths is touched.

## Development

```sh
git clone https://github.com/mestump/PlaylistGen && cd PlaylistGen
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

## Version history

- **v1** (archived on the
  [`v1-legacy`](https://github.com/mestump/PlaylistGen/tree/v1-legacy) branch):
  iTunes-XML pipeline with optional Last.fm/AI enrichment. Retired.
- **v2** (this): greenfield rewrite — local-first, event-store based, no
  account or API key required for the core loop.

## License

[MIT](LICENSE)
