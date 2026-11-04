"""Playlist generation v1: coherence-chained greedy over the candidate pool.

v0 lesson (2026-10-04): top-N by static score interleaves unrelated styles --
a cast recording lands mid-post-hardcore. v1 builds a sequence: each next
track must cohere with the running window (genre jaccard + feature cosine +
affinity pmi) or it opens a *section break* (capped, reported).

Static score as v0: affinity mass + popularity + centroid proximity, skip-
penalized. Deterministic per seed. Quality report printed every run.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import time
from pathlib import Path

EXT_RANK = {".flac": 0, ".wav": 1, ".aiff": 2, ".m4a": 3, ".ogg": 4, ".opus": 4, ".mp3": 5}

SEEDS = 40
N = 30
ARTIST_CAP = 3
HOPS = 2
DECAY = 0.5
FAMILIAR = 0.4
SKIP_PENALTY = 0.5
W_AFF, W_POP, W_FEAT = 0.55, 0.25, 0.20

W_COH = 0.45  # weight of coherence vs static in chain step
W_GENRE, W_FEATC, W_AFFC = 0.45, 0.35, 0.20
COH_MIN = 0.35
MAX_BREAKS = 4
WINDOW = 3
FEAT_DIMS = ("energy", "danceability", "valence", "acousticness", "instrumentalness", "speechiness", "liveness")


def _pick_path(con: sqlite3.Connection, tid: int) -> str | None:
    rows = con.execute("SELECT path, size FROM track_paths WHERE track_id=?", (tid,)).fetchall()
    if not rows:
        return None
    rows.sort(key=lambda r: (EXT_RANK.get(Path(r["path"]).suffix.lower(), 9), -(r["size"] or 0)))
    return rows[0]["path"]


def _load_json(con: sqlite3.Connection, tid: int) -> dict | None:
    row = con.execute(
        "SELECT scalars FROM features WHERE track_id=? AND source='spotify-parquet'", (tid,)
    ).fetchone()
    return json.loads(row["scalars"]) if row else None


def _feature_centroid(con: sqlite3.Connection, seed_ids: list[int]) -> dict[str, float]:
    acc: dict[str, list[float]] = {}
    for tid in seed_ids:
        sc = _load_json(con, tid) or {}
        for k in FEAT_DIMS:
            v = sc.get(k)
            if isinstance(v, (int, float)):
                acc.setdefault(k, []).append(float(v))
    return {k: sum(v) / len(v) for k, v in acc.items() if v}


def _centroid_dist(sc: dict, centroid: dict[str, float]) -> float | None:
    if not centroid:
        return None
    num = den = 0.0
    for k, c in centroid.items():
        v = sc.get(k)
        if not isinstance(v, (int, float)):
            continue
        num += (float(v) - c) ** 2
        den += 1
    return math.sqrt(num / den) if den else None


def _zvec(sc: dict, mu: dict[str, float], sd: dict[str, float]) -> list[float]:
    v = []
    for k in FEAT_DIMS:
        x = sc.get(k)
        x = float(x) if isinstance(x, (int, float)) else mu.get(k, 0.0)
        s = sd.get(k) or 1.0
        v.append((x - mu.get(k, 0.0)) / s)
    return v


def _cos(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return num / (na * nb)


def _jack(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.5  # neutral when unknown
    u = len(a | b)
    return len(a & b) / u if u else 0.5


def generate(con: sqlite3.Connection, n: int = N, seed: int = 0, scene: str | None = None) -> dict:
    from .db import ensure_derived

    ensure_derived(con)
    rng = random.Random(seed)
    spoken = {r["track_id"] for r in con.execute("SELECT track_id FROM track_signal WHERE is_spoken>0")}

    genres: dict[int, set[str]] = {}
    for tid, g in con.execute("SELECT track_id, genre FROM track_genre"):
        genres.setdefault(tid, set()).add(g)

    weights = {}
    for r in con.execute(
        "SELECT track_id, COALESCE(plays_total,0) pt, COALESCE(plays_events,0) pe FROM track_signal"
    ):
        w = r["pt"] + 2 * r["pe"]
        if w > 0 and r["track_id"] not in spoken:
            weights[r["track_id"]] = math.log1p(w)
    seed_pool = sorted(weights, key=lambda t: -weights[t])
    scene_set: set[str] = set()
    if scene:
        if "," in scene:
            tokens = {t.strip().lower() for t in scene.split(",") if t.strip()}
            scene_set = {g for g in {x for gs in genres.values() for x in gs} if g.lower() in tokens}
        else:
            s = scene.lower()
            scene_set = {g for g in {x for gs in genres.values() for x in gs} if s in g}
        scene_tracks = [t for t, gs in ((t, genres.get(t) or set()) for t in weights) if gs & scene_set]
        seed_pool = sorted(scene_tracks, key=lambda t: -weights[t])
    seed_set = set(seed_pool[:SEEDS])

    aff: dict[int, dict[int, float]] = {}
    for a, b, pmi in con.execute("SELECT a,b,pmi FROM track_affinity"):
        aff.setdefault(a, {})[b] = pmi
        aff.setdefault(b, {})
    mass: dict[int, float] = {}
    frontier = {t: weights[t] for t in seed_set}
    for hop in range(HOPS):
        nxt: dict[int, float] = {}
        for node, w in frontier.items():
            for nb, pmi in aff.get(node, {}).items():
                if nb in spoken:
                    continue
                contrib = w * math.exp(min(pmi, 10.0)) * (DECAY**hop)
                nxt[nb] = nxt.get(nb, 0.0) + contrib
        for t, v in nxt.items():
            mass[t] = mass.get(t, 0.0) + v
        frontier = nxt
    mmax = max(mass.values(), default=1.0) or 1.0

    have_path = {r["track_id"] for r in con.execute("SELECT DISTINCT track_id FROM track_paths")}

    # candidate pool + feature stats for z-scoring
    wmax = max(weights.values()) if weights else 1.0
    centroid = _feature_centroid(con, sorted(seed_set))
    pool: dict[int, dict] = {}
    scal_pool: dict[int, dict] = {}
    for t, m in mass.items():
        if t not in have_path:
            continue
        if scene_set and not (genres.get(t, set()) & scene_set) and not (
            set(aff.get(t, {})) & seed_set
        ):
            continue  # scene mode: genre tag or a direct affinity bridge required
        sig = con.execute(
            "SELECT plays_total, skip_ratio FROM track_signal WHERE track_id=?", (t,)
        ).fetchone()
        sc = _load_json(con, t) or {}
        pt = sig["plays_total"] if sig else 0
        sr = sig["skip_ratio"] if sig and sig["skip_ratio"] is not None else 0.0
        aff_n = m / mmax
        pop_n = math.log1p(pt) / math.log1p(max(wmax, 1))
        d = _centroid_dist(sc, centroid)
        feat_n = 1.0 - min((d if d is not None else 1.0) / 0.5, 1.0)
        s = W_AFF * aff_n + W_POP * pop_n + W_FEAT * feat_n
        s *= 1.0 - SKIP_PENALTY * min(sr, 1.0)
        pool[t] = {"static": s, "seed": t in seed_set, "sc": sc, "aff": aff.get(t, {})}
        scal_pool[t] = sc

    if not pool:
        # Cold start: no listening history anywhere. Every pathed, non-spoken
        # track is eligible; ranking falls to genre/feature coherence only.
        for t in sorted(have_path):
            if t in spoken:
                continue
            if scene_set and not (genres.get(t, set()) & scene_set):
                continue
            sc = _load_json(con, t) or {}
            pool[t] = {"static": 1.0, "seed": False, "sc": sc, "aff": {}}
            scal_pool[t] = sc
    mu, sd = _pool_stats(scal_pool)
    z = {t: _zvec(d["sc"], mu, sd) for t, d in pool.items()}

    def coh(a: int, b: int) -> float:
        g = _jack(genres.get(a) or set(), genres.get(b) or set())
        c = _cos(z[a], z[b]) if a in z and b in z else 0.5
        p = pool.get(a, {}).get("aff", {}).get(b)
        p01 = min(p, 6.0) / 6.0 if p is not None else 0.0
        return W_GENRE * g + W_FEATC * (0.5 + 0.5 * c) + W_AFFC * p01

    n_familiar = round(n * FAMILIAR)
    picked: list[int] = []
    fam = nov = 0
    breaks = 0
    artist_count: dict[str, int] = {}
    artist_of = {r["id"]: r["artist_canon"] for r in con.execute("SELECT id, artist_canon FROM tracks")}

    while len(picked) < n:
        ok = [
            t
            for t in pool
            if t not in picked
            and artist_count.get(artist_of[t], 0) < ARTIST_CAP
            and not (seed_set and pool[t]["seed"] and fam >= n_familiar)
            and not (seed_set and not pool[t]["seed"] and nov >= n - n_familiar)
        ]
        if not ok:
            break
        window = picked[-WINDOW:]
        best_t, best_v, best_coh = None, -1.0, 0.0
        for t in ok:
            c = max((coh(t, w) for w in window), default=1.0)
            v = (1 - W_COH) * pool[t]["static"] + W_COH * c
            if v > best_v + 1e-12 or (abs(v - best_v) < 1e-12 and rng.random() < 0.5):
                best_t, best_v, best_coh = t, v, c
        if picked and best_coh < COH_MIN and breaks < MAX_BREAKS:
            # section break: pure static best
            best_t = max(ok, key=lambda t: (pool[t]["static"], -t))
            breaks += 1
        if best_t is None:
            break
        picked.append(best_t)
        if pool[best_t]["seed"]:
            fam += 1
        else:
            nov += 1
        artist_count[artist_of[best_t]] = artist_count.get(artist_of[best_t], 0) + 1

    lines = ["#EXTM3U"]
    written = 0
    for t in picked:
        p = _pick_path(con, t)
        if not p or not Path(p).exists():
            continue
        r = con.execute("SELECT artist, title FROM tracks WHERE id=?", (t,)).fetchone()
        lines += [f"#EXTINF:-1,{r['artist']} - {r['title']}", p]
        written += 1

    out = Path.home() / ".local/share/pgen/playlists"
    out.mkdir(parents=True, exist_ok=True)
    slug = scene.lower().replace(",", "-").replace(" ", "-").strip("-")[:24] if scene else "all"
    path = out / f"daily-{time.strftime('%Y%m%d-%H%M%S')}-s{seed}-{slug}.m3u"
    path.write_text("\n".join(lines) + "\n")

    # quality report
    if len(picked) > 1:
        seq_coh = [coh(picked[i + 1], picked[i]) for i in range(len(picked) - 1)]
        mean_coh = sum(seq_coh) / len(seq_coh)
        low = sum(1 for c in seq_coh if c < COH_MIN)
    else:
        mean_coh, low = 1.0, 0
    hist: dict[str, int] = {}
    for t in picked:
        hist[artist_of[t]] = hist.get(artist_of[t], 0) + 1
    report = {
        "mean_seq_coherness": round(mean_coh, 3),
        "below_threshold_links": low,
        "section_breaks": breaks,
        "artist_hist": dict(sorted(hist.items(), key=lambda kv: -kv[1])),
        "familiar": fam,
    }
    con.execute(
        f"INSERT INTO runs(kind,seed,params,m3u_path) VALUES('daily-{'scene' if scene else 'v1'}',?,?,?)",
        (
            seed,
            json.dumps(
                {
                    "seeds": SEEDS,
                    "n": n,
                    "hops": HOPS,
                    "w_static": [W_AFF, W_POP, W_FEAT],
                    "w_coh": [W_GENRE, W_FEATC, W_AFFC],
                    "coh_min": COH_MIN,
                    **report,
                }
            ),
            str(path),
        ),
    )
    con.commit()
    return {"path": str(path), "picked": written, "candidates": len(pool), "report": report}


def _pool_stats(pool_scalars: dict[int, dict]) -> tuple[dict[str, float], dict[str, float]]:
    mu: dict[str, float] = {}
    sd: dict[str, float] = {}
    for k in FEAT_DIMS:
        vals = [float(d[k]) for d in pool_scalars.values() if isinstance(d.get(k), (int, float))]
        if not vals:
            mu[k], sd[k] = 0.0, 1.0
            continue
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        mu[k], sd[k] = m, math.sqrt(var) or 1.0
    return mu, sd


def main(argv=None) -> int:
    from .db import connect

    ap = argparse.ArgumentParser(prog="pgen-make")
    ap.add_argument("--db", default=str(Path.home() / ".local/share/pgen/v2.db"))
    ap.add_argument("-n", type=int, default=N)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scene", default=None, help="genre token, e.g. 'hardcore', 'country', 'folk'")
    args = ap.parse_args(argv)
    con = connect(args.db)
    res = generate(con, args.n, args.seed, args.scene)
    print(json.dumps(res, indent=2))
    con.close()
    return 0 if res["picked"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
