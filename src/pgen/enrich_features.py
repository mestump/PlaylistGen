"""M2 stage 1: scalar features via spotify-parquet match (match-first, per design §6.1).

Memory-safe shape (the 38GB/256M-row parquet is streamed, never materialized):
  1. libk = the ~34k known (artist,title)-keys  -> tiny hash-build side
  2. ONE scan of tracks/artists/track_artists with the join pushed down -> matches (~100k rows)
  3. best-row-per-tid by popularity -> ~30k spotify_ids
  4. ONE scan of track_audio_features joined on those ids -> scalars
The tid->spotify_id match table is cached to ~/pgen-tmp/match-cache.parquet so the
expensive scan happens at most once ever; features land in sqlite (idempotent).

Usage: python -m pgen.enrich_features [--limit N] [--rebuild] [--threads 8] [--mem-limit 4GB]
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path

PD = "/mnt/storage/spotify-data"
CACHE = Path.home() / "pgen-tmp/match-cache.parquet"

_ACC = str.maketrans(
    "áàâäãåéèêëíìîïóòôöõúùûüñç",
    "aaaaaaeeeeiiiiooooouuuunc",
)
re_paren = re.compile(r"\([^)]*\)|\[[^\]]*\]")
re_nonalnum = re.compile(r"[^a-z0-9]")


def norm(s: str) -> str:
    s = s.lower().translate(_ACC)
    s = re_paren.sub(" ", s)
    return re_nonalnum.sub("", s)


def _norm_sql(col: str) -> str:
    return (
        f"regexp_replace(lower(translate(regexp_replace(regexp_replace({col}, '\\([^)]*\\)', ' '), "
        f"'\\[[^\\]]*\\]', ' '), 'áàâäãåéèêëíìîïóòôöõúùûüñç', 'aaaaaaeeeeiiiiooooouuuunc')), "
        f"'[^a-z0-9]', '', 'g')"
    )


def run(db_path: str, limit: int | None, rebuild: bool, threads: int, mem_limit: str) -> int:
    import duckdb

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, artist, title FROM tracks WHERE artist IS NOT NULL AND TRIM(artist)<>'' "
        "AND title IS NOT NULL AND TRIM(title)<>''"
    ).fetchall()
    total_all = len(rows)
    if not rebuild:
        have = {r[0] for r in con.execute("SELECT track_id FROM features WHERE source='spotify-parquet'")}
        rows = [r for r in rows if r["id"] not in have]
    if limit:
        rows = rows[:limit]
    if not rows:
        print("nothing to enrich")
        return 0

    keyed: dict[tuple[str, str], int] = {}
    for r in rows:
        k = norm(r["artist"]), norm(r["title"])
        if k[0] and k[1]:
            keyed.setdefault(k, r["id"])
    print(f"keys to match: {len(keyed):,} (library identity rows {total_all:,})", flush=True)

    d = duckdb.connect()
    d.execute(f"PRAGMA threads={threads}")
    d.execute(f"PRAGMA memory_limit='{mem_limit}'")
    spill = Path.home() / "pgen-tmp/duck-spill"
    spill.mkdir(parents=True, exist_ok=True)
    d.execute(f"PRAGMA temp_directory='{spill}'")

    # ---- stage A: identity match (cached) ------------------------------------
    need_match = True
    if CACHE.exists() and not rebuild:
        try:
            cached = d.sql(f"SELECT DISTINCT tid FROM read_parquet('{CACHE}')").df()
            need = {tid for _, tid in keyed.items()} - set(cached["tid"])
            need_match = len(need) > 0
        except Exception:
            need_match = True

    if need_match:
        d.execute("CREATE OR REPLACE TABLE libk(artist VARCHAR, title VARCHAR, tid BIGINT)")
        d.executemany("INSERT INTO libk VALUES(?,?,?)", [(a, b, i) for (a, b), i in keyed.items()])
        print("scanning parquet for identity match (single pass)...", flush=True)
        matches = d.sql(f"""
        WITH t AS (
          SELECT rowid, id, popularity, {_norm_sql('name')} AS tn
          FROM read_parquet('{PD}/tracks.parquet', union_by_name=true) WHERE name IS NOT NULL
        ),
        an AS (
          SELECT rowid, {_norm_sql('name')} AS an
          FROM read_parquet('{PD}/artists.parquet', union_by_name=true) WHERE name IS NOT NULL
        ),
        ta AS (SELECT DISTINCT track_rowid, artist_rowid FROM read_parquet('{PD}/track_artists.parquet'))
        SELECT DISTINCT k.tid, t.id AS spotify_id, t.popularity
        FROM libk k
        JOIN t ON t.tn = k.title
        JOIN ta ON ta.track_rowid = t.rowid
        JOIN an ON an.rowid = ta.artist_rowid AND an.an = k.artist
        """).df()
        n_matched = matches["tid"].nunique()
        print(f"match: {n_matched:,}/{len(keyed):,} keys ({100*n_matched/len(keyed):.1f}%)", flush=True)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        if CACHE.exists():  # merge: cached union grows across runs
            d.register("new_matches", matches[["tid", "spotify_id", "popularity"]])
            merged = d.sql(f"""
                SELECT tid, spotify_id, max(popularity) AS popularity FROM (
                  SELECT tid, spotify_id, popularity FROM read_parquet('{CACHE}')
                  UNION ALL
                  SELECT tid, spotify_id, popularity FROM new_matches
                ) GROUP BY tid, spotify_id
            """).df()
            d.register("merged_matches", merged)
            d.sql(f"COPY merged_matches TO '{CACHE}' (FORMAT PARQUET, COMPRESSION zstd)")
        else:
            d.register("matches_df", matches)
            d.sql(f"COPY matches_df TO '{CACHE}' (FORMAT PARQUET, COMPRESSION zstd)")
    else:
        matches = d.sql(f"SELECT tid, any_value(spotify_id) spotify_id, max(popularity) popularity FROM read_parquet('{CACHE}') GROUP BY tid, spotify_id").df()
        n_matched = matches["tid"].nunique()
        print(f"match (cached): {n_matched:,} keys", flush=True)

    # ---- stage B: best row per tid, then features ----------------------------
    d.register("matches", matches[["tid", "spotify_id", "popularity"]])
    best = d.sql("""
        SELECT tid, spotify_id, popularity FROM (
          SELECT *, row_number() OVER (PARTITION BY tid ORDER BY popularity DESC) rn
          FROM matches) WHERE rn=1
    """).df()
    print(f"canonical rows: {len(best):,}", flush=True)
    d.register("best_ids", best[["spotify_id"]])

    feats = d.sql(f"""
        SELECT f.track_id,
               avg(TRY_CAST(f.tempo AS DOUBLE)) tempo,
               avg(TRY_CAST(f.energy AS DOUBLE)) energy,
               avg(TRY_CAST(f.danceability AS DOUBLE)) danceability,
               avg(TRY_CAST(f.valence AS DOUBLE)) valence,
               avg(TRY_CAST(f.loudness AS DOUBLE)) loudness,
               avg(TRY_CAST(f.speechiness AS DOUBLE)) speechiness,
               avg(TRY_CAST(f.acousticness AS DOUBLE)) acousticness,
               avg(TRY_CAST(f.instrumentalness AS DOUBLE)) instrumentalness,
               avg(TRY_CAST(f.liveness AS DOUBLE)) liveness,
               mode(TRY_CAST(f.key AS INTEGER)) AS key_mode,
               mode(TRY_CAST(f.mode AS INTEGER)) AS mode_mode,
               avg(TRY_CAST(f.time_signature AS DOUBLE)) time_signature
        FROM read_parquet('{PD}/track_audio_features.parquet', union_by_name=true) f
        JOIN best_ids b ON b.spotify_id = f.track_id
        GROUP BY f.track_id
    """).df()
    print(f"feature rows: {len(feats):,}", flush=True)

    pop = dict(zip(best["tid"], best["popularity"]))
    fmap = {r.track_id: r for r in feats.itertuples(index=False)}
    written = 0
    for tid, spotify_id in zip(best["tid"], best["spotify_id"]):
        fr = fmap.get(spotify_id)
        if fr is None:
            continue
        scal = {
            "tempo": round(float(fr.tempo), 2) if fr.tempo is not None else None,
            "energy": round(float(fr.energy), 3) if fr.energy is not None else None,
            "danceability": round(float(fr.danceability), 3) if fr.danceability is not None else None,
            "valence": round(float(fr.valence), 3) if fr.valence is not None else None,
            "loudness": round(float(fr.loudness), 2) if fr.loudness is not None else None,
            "speechiness": round(float(fr.speechiness), 3) if fr.speechiness is not None else None,
            "acousticness": round(float(fr.acousticness), 3) if fr.acousticness is not None else None,
            "instrumentalness": round(float(fr.instrumentalness), 3) if fr.instrumentalness is not None else None,
            "liveness": round(float(fr.liveness), 3) if fr.liveness is not None else None,
            "key": int(fr.key_mode) if fr.key_mode is not None else None,
            "mode": int(fr.mode_mode) if fr.mode_mode is not None else None,
            "time_signature": float(fr.time_signature) if fr.time_signature is not None else None,
            "popularity": int(pop[tid]) if pop.get(tid) is not None else None,
        }
        con.execute(
            "INSERT INTO features(track_id,model,dim,vec,scalars,source) "
            "VALUES(?,NULL,NULL,NULL,?,'spotify-parquet') "
            "ON CONFLICT(track_id) DO UPDATE SET scalars=excluded.scalars, "
            "source='spotify-parquet', updated=datetime('now')",
            (int(tid), json.dumps(scal)),
        )
        written += 1
    con.commit()
    d.close()
    featn = con.execute("SELECT COUNT(*) FROM features WHERE source='spotify-parquet'").fetchone()[0]
    print(f"written this run: {written:,}", flush=True)
    print(f"coverage: {featn:,}/{total_all:,} = {100*featn/max(total_all,1):.1f}%")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pgen-enrich-features")
    ap.add_argument("--db", default=str(Path.home() / ".local/share/pgen/v2.db"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--mem-limit", default="3GB")
    args = ap.parse_args(argv)
    return run(args.db, args.limit, args.rebuild, args.threads, args.mem_limit)


if __name__ == "__main__":
    raise SystemExit(main())
