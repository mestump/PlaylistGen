"""Unit tests for the v1 coherence-chain generator (synthetic DB, no real data)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from pgen.db import SCHEMA
from pgen.generate import generate


@pytest.fixture()
def db(tmp_path: Path):
    con = sqlite3.connect(tmp_path / "t.db")
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    con.execute(
        "CREATE TABLE IF NOT EXISTS track_signal("
        "track_id INTEGER PRIMARY KEY, plays_90d REAL, plays_365d REAL, plays_events INTEGER,"
        "plays_counters INTEGER, plays_total REAL, skips INTEGER, skip_ratio REAL,"
        "last_played_ts INTEGER, days_since_play REAL, artist_key TEXT,"
        "artist_plays_365d REAL, artist_rank INTEGER, hour_peak INTEGER, is_spoken INTEGER DEFAULT 0)"
    )
    # 12 tracks: artists A/B (post-hardcore scene), C (musicals), +1 spoken
    tid = {}
    counters: dict[str, int] = {}
    for artist, base in [("A", 4), ("B", 4), ("C", 3), ("S", 1)]:
        for i in range(base):
            title = f"{artist.lower()}{i}"
            id_ = con.execute(
                "INSERT INTO tracks(artist_canon,title_canon,artist,title) VALUES(?,?,?,?)",
                (artist.lower(), title, artist, title),
            ).lastrowid
            tid[f"{artist}{i}"] = id_
    for key, id_ in tid.items():
        p = tmp_path / f"{key}.flac"
        p.write_bytes(b"x")
        con.execute(
            "INSERT INTO track_paths(track_id,path,source) VALUES(?,?,'test')", (id_, str(p))
        )
        con.execute(
            "INSERT INTO track_signal(track_id,plays_total,plays_events,is_spoken) VALUES(?,?,?,?)",
            (id_, 10 + id_, 0, 1 if key == "S" else 0),
        )
    con.execute(
        "CREATE TABLE IF NOT EXISTS track_genre("
        "track_id INTEGER NOT NULL, genre TEXT NOT NULL, PRIMARY KEY(track_id,genre))"
    )
    for k, g in [("a", "post-hardcore"), ("b", "post-hardcore"), ("c", "musicals")]:
        for id_, in con.execute("SELECT id FROM tracks WHERE artist_canon=?", (k,)):
            con.execute("INSERT INTO track_genre VALUES(?,?)", (id_, g))
    # affinity: A<->B strong, C isolated
    con.execute("CREATE TABLE IF NOT EXISTS track_affinity(a INTEGER, b INTEGER, co INTEGER, pmi REAL)")
    con.execute("CREATE TABLE IF NOT EXISTS track_genre(track_id INTEGER, genre TEXT, PRIMARY KEY(track_id,genre))")
    a1 = con.execute("SELECT id FROM tracks WHERE title_canon='a0'").fetchone()[0]
    a2 = con.execute("SELECT id FROM tracks WHERE title_canon='a1'").fetchone()[0]
    b0 = con.execute("SELECT id FROM tracks WHERE title_canon='b0'").fetchone()[0]
    b1 = con.execute("SELECT id FROM tracks WHERE title_canon='b1'").fetchone()[0]
    for x, y in ((a1, b0), (a1, b1), (a2, b0)):
        con.execute("INSERT INTO track_affinity VALUES(?,?,?,?)", (min(x, y), max(x, y), 5, 3.0))
    scal = json.dumps({"energy": 0.8, "valence": 0.5})
    for key in ("A0", "A1", "B0", "B1"):
        con.execute(
            "INSERT INTO features(track_id,scalars,source) VALUES(?,?,'spotify-parquet')",
            (tid[key], scal),
        )
    con.commit()
    yield con
    con.close()


def test_scene_excludes_other_genre(db, tmp_path):
    res = generate(db, n=6, seed=0, scene="hardcore")
    rows = [ln for ln in Path(res["path"]).read_text().splitlines() if ln.startswith("#EXTINF")]
    assert rows, "playlist empty"
    artists = {ln.split(",")[1].split(" - ")[0] for ln in rows}
    assert "C" not in artists and "S" not in artists  # musicals + spoken excluded
    assert res["report"]["artist_hist"]  # histogram present


def test_deterministic_same_seed(db: sqlite3.Connection, tmp_path: Path):
    r1 = generate(db, n=5, seed=3)
    r2 = generate(db, n=5, seed=3)
    body1 = Path(r1["path"]).read_text().splitlines()[1:]
    body2 = Path(r2["path"]).read_text().splitlines()[1:]
    assert body1 == body2
    assert len(body1) % 2 == 0 and len(body1) // 2 >= 1


def test_artist_cap_and_runs_logged(db: sqlite3.Connection):
    res = generate(db, n=6, seed=1)
    hist = res["report"]["artist_hist"]
    assert all(v <= 3 for v in hist.values())
    row = db.execute("SELECT kind, m3u_path FROM runs ORDER BY id DESC LIMIT 1").fetchone()
    assert row["m3u_path"] == res["path"]


def test_cold_start_no_signal_data(tmp_path: Path):
    """Fresh store: tracks+paths only, no build ran, no listening history."""
    con = sqlite3.connect(tmp_path / "cold.db")
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    for artist, n in [("A", 4), ("B", 4), ("C", 4)]:
        for i in range(n):
            id_ = con.execute(
                "INSERT INTO tracks(artist_canon,title_canon,artist,title) VALUES(?,?,?,?)",
                (artist.lower(), f"{artist.lower()}{i}", artist, f"{artist.lower()}{i}"),
            ).lastrowid
            p = tmp_path / f"{artist.lower()}{i}.flac"
            p.write_bytes(b"x")
            con.execute("INSERT INTO track_paths(track_id,path,source) VALUES(?,?,'test')", (id_, str(p)))
    con.commit()
    res = generate(con, n=6, seed=0)
    rows = [ln for ln in Path(res["path"]).read_text().splitlines() if ln.startswith("#EXTINF")]
    assert len(rows) == 6, f"cold start picked {len(rows)}"
    hist = res["report"]["artist_hist"]
    assert all(v <= 3 for v in hist.values())
    con.close()


def test_cold_start_scene_no_genres(tmp_path: Path):
    """Scene mode on a store with no genre table rows exits cleanly, not with a crash."""
    con = sqlite3.connect(tmp_path / "cold2.db")
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    id_ = con.execute(
        "INSERT INTO tracks(artist_canon,title_canon,artist,title) VALUES('a','a0','A','a0')"
    ).lastrowid
    p = tmp_path / "a0.flac"
    p.write_bytes(b"x")
    con.execute("INSERT INTO track_paths(track_id,path,source) VALUES(?,?,'test')", (id_, str(p)))
    con.commit()
    res = generate(con, n=5, seed=0, scene="ambient")  # no genres -> empty pool allowed
    assert "picked" in res
    con.close()
