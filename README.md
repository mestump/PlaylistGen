# playlistgen-v2

Local-library playlist generation from real listening history. Greenfield successor to
[mestump/PlaylistGen](https://github.com/mestump/PlaylistGen) (v1 evaluation: 2026-10-04, retired).

Design doc: `~/Downloads/playlistgen-v2-design.md` (mirrored in repo docs/ once finalized).

## Status: M0/M1 — ingest layer, real data

| M | scope | state |
|---|---|---|
| M0 | package, CLI, schema, CI-able tests | done |
| M1 | ingest: files / strawberry / ipod / spotify-history + identity resolution | done, validated on real sources |
| M2 | features (parquet match-first + audio embeddings), signals materialized | next |
| M3 | `pgen make daily` + explanations, seeded | |
| M4 | all playlist types, feedback loop | |
| M5 | polish, cron, docs | |

## Install

```sh
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
```

## Ingest

```sh
pgen ingest files                 # /mnt/nas_media/Music (canonical library)
pgen ingest strawberry            # ~/.local/share/strawberry/.../strawberry.db (read-only)
pgen ingest ipod                  # autodetects /run/media/*/*/iPod_Control/iTunes/iTunesDB
pgen ingest spotify-history --spotify-glob ~/sync/Streaming_History_Audio_*.json
pgen status
```

All sources are idempotent (stamped by mtime). Sources:

- **files** — walk library, tags via mutagen, `Artist - Title` filename fallback
- **strawberry** — counters to `source_counts` (no fabricated events)
- **ipod** — iTunesDB parsed directly; device files are authoritative, Strawberry's
  device cache is stale by observation. Playcount offset `mhit+0x30` (monotone-validated);
skipcount `+0x2C` provisional. See module docstring in `pgen/ingest/ipod.py`.
- **spotify-history** — Extended Streaming History JSON (2012–2025 shards)

## Store

Single sqlite (WAL) at `~/.local/share/pgen/v2.db`. Append-only `play_events` +
per-source `source_counts`; `tracks` keyed by `(artist_canon, title_canon)` — see
`pgen/normalize.py`. Everything except events/counts is a rebuildable projection.

## Notes

- Player DBs are opened read-only; v2 never writes to them.
- `/mnt/nas_media` is autofs — first access stalls while it mounts.
- Spotify scalar features come from the local parquet dump (`/mnt/storage/spotify-data`,
  38GB) via duckdb in M2 — match-first, local inference only for the residue.
