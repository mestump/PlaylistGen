"""Playlist generation v0: seeded affinity-graph walk + feature centricity.

Inputs: track_signal (plays/skips/spoken), track_affinity (pmi), features (scalars).
v0 recipe (daily-mix shape):
  seeds   = top SEEDS tracks by blended play weight (spoken excluded)
  spread  = 2-hop personalized-PMI mass over track_affinity from seeds
  score   = 0.55*affinity + 0.25*log1p(plays_total) + 0.20*feature_centrality
            penalized by skip_ratio
  output  = N tracks, <=ARTIST_CAP per artist, ~FAMILIAR fraction seeds,
            paths resolved from track_paths (flac>ogg>m4a>mp3, largest file)
Deterministic given seed. No model files, no embeddings yet -- v0 is the
baseline every later idea must beat.
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


def _pick_path(con: sqlite3.Connection, tid: int) -> str | None:
    rows = con.execute("SELECT path, size FROM track_paths WHERE track_id=?", (tid,)).fetchall()
    if not rows:
        return None
    rows.sort(key=lambda r: (EXT_RANK.get(Path(r["path"]).suffix.lower(), 9), -(r["size"] or 0)))
    return rows[0]["path"]


def _feature_centroid(con: sqlite3.Connection, seed_ids: list[int]) -> dict[str, float]:
    import json as _json

    acc: dict[str, list[float]] = {}
    for tid in seed_ids:
        row = con.execute(
            "SELECT scalars FROM features WHERE track_id=? AND source='spotify-parquet'", (tid,)
        ).fetchone()
        if not row:
            continue
        for k, v in _json.loads(row["scalars"]).items():
            if isinstance(v, (int, float)):
                acc.setdefault(k, []).append(float(v))
    return {k: sum(v) / len(v) for k, v in acc.items() if v}


def _feature_dist(con: sqlite3.Connection, tid: int, centroid: dict[str, float]) -> float | None:
    row = con.execute(
        "SELECT scalars FROM features WHERE track_id=? AND source='spotify-parquet'", (tid,)
    ).fetchone()
    if not row or not centroid:
        return None
    import json as _json

    sc = _json.loads(row["scalars"])
    num = den = 0.0
    for k, c in centroid.items():
        v = sc.get(k)
        if not isinstance(v, (int, float)):
            continue
        num += (float(v) - c) ** 2
        den += 1
    return math.sqrt(num / den) if den else None


def generate(con: sqlite3.Connection, n: int = N, seed: int = 0) -> dict:
    rng = random.Random(seed)
    spoken = {r["track_id"] for r in con.execute("SELECT track_id FROM track_signal WHERE is_spoken>0")}
    have_feat = {r["track_id"] for r in con.execute("SELECT track_id FROM features WHERE source='spotify-parquet'")}

    # blended weight: counters + events, events count double (timestamped truth)
    weights = {}
    for r in con.execute(
        "SELECT track_id, COALESCE(plays_total,0) pt, COALESCE(plays_events,0) pe FROM track_signal"
    ):
        w = r["pt"] + 2 * r["pe"]
        if w > 0 and r["track_id"] not in spoken:
            weights[r["track_id"]] = math.log1p(w)
    seeds = sorted(weights, key=lambda t: -weights[t])[:SEEDS]

    # 2-hop personalized PMI mass
    aff: dict[int, dict[int, float]] = {}
    for a, b, pmi in con.execute("SELECT a,b,pmi FROM track_affinity"):
        aff.setdefault(a, {})[b] = pmi
        aff.setdefault(b, {}).setdefault(a, 0.0)
        if aff[b].get(a, 0) < pmi:
            pass  # symmetric already ensured at build

    mass: dict[int, float] = {}
    frontier = {t: weights[t] for t in seeds}
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

    centroid = _feature_centroid(con, seeds)
    wmax = max(weights.values()) if weights else 1.0
    have_path = {r["track_id"] for r in con.execute("SELECT DISTINCT track_id FROM track_paths")}

    scored = []
    for t, m in mass.items():
        if t not in have_path:
            continue
        sig = con.execute(
            "SELECT plays_total, skip_ratio FROM track_signal WHERE track_id=?", (t,)
        ).fetchone()
        pt = sig["plays_total"] if sig else 0
        sr = sig["skip_ratio"] if sig and sig["skip_ratio"] is not None else 0.0
        aff_n = m / mmax
        pop_n = math.log1p(pt) / math.log1p(max(wmax, 1))
        d = _feature_dist(con, t, centroid)
        feat_n = 1.0 - min((d or 1.0) / 0.5, 1.0)  # 0 dist -> 1.0
        s = W_AFF * aff_n + W_POP * pop_n + W_FEAT * feat_n
        s *= 1.0 - SKIP_PENALTY * min(sr, 1.0)
        familiar = 1.0 if t in seeds else 0.0
        scored.append((s, t, familiar))
    scored.sort(key=lambda x: (-x[0], rng.random()))

    n_familiar = round(n * FAMILIAR)
    picked: list[int] = []
    fam = nov = 0
    artist_count: dict[str, int] = {}
    for s, t, is_seed in scored:
        if len(picked) >= n:
            break
        if is_seed and fam >= n_familiar:
            continue
        if not is_seed and nov >= n - n_familiar:
            continue
        a = con.execute("SELECT artist_canon FROM tracks WHERE id=?", (t,)).fetchone()["artist_canon"]
        if artist_count.get(a, 0) >= ARTIST_CAP:
            continue
        artist_count[a] = artist_count.get(a, 0) + 1
        picked.append(t)
        if is_seed:
            fam += 1
        else:
            nov += 1

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
    path = out / f"daily-{time.strftime('%Y%m%d-%H%M%S')}.m3u"
    path.write_text("\n".join(lines) + "\n")

    import json as _json

    con.execute(
        "INSERT INTO runs(kind,seed,params,m3u_path) VALUES('daily-v0',?,?,?)",
        (
            seed,
            _json.dumps({"seeds": SEEDS, "n": n, "hops": HOPS, "w": [W_AFF, W_POP, W_FEAT]}),
            str(path),
        ),
    )
    con.commit()
    return {
        "path": str(path),
        "picked": written,
        "candidates": len(scored),
    }


def main(argv=None) -> int:
    from .db import connect

    ap = argparse.ArgumentParser(prog="pgen-make")
    ap.add_argument("--db", default=str(Path.home() / ".local/share/pgen/v2.db"))
    ap.add_argument("-n", type=int, default=N)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    con = connect(args.db)
    res = generate(con, args.n, args.seed)
    print(json.dumps(res, indent=2))
    con.close()
    return 0 if res["picked"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
