"""M3 stage 1: sessionize listen events and materialize track co-play affinity.

Session: consecutive plays <30min apart on the merged global timeline.
Affinity: PMI over session co-play, counts smoothed, symmetrized.
Output tables:
  listen_session(session_id INTEGER, started INTEGER, n_tracks INTEGER)
  play_events.session_id (backfill)
  track_affinity(a INT, b INT, co INT, pmi REAL)  -- a<b, top-K per track kept
"""

from __future__ import annotations

import argparse
import math
import sqlite3
from pathlib import Path

SESSION_GAP = 1800  # 30min
PMI_MIN_CO = 2  # ignore pairs seen once
TOPK = 200  # keep top-K partners per track


def sessionize(con: sqlite3.Connection) -> int:
    con.execute("DROP TABLE IF EXISTS listen_session")
    con.executescript(
        """
        CREATE TABLE listen_session(
          session_id INTEGER PRIMARY KEY,
          started INTEGER, n_tracks INTEGER);
        """
    )
    con.execute("UPDATE play_events SET session_id=NULL")
    # order events, mark session starts on >GAP gaps, cumsum -> session id
    con.execute(
        """
        WITH ord AS (
          SELECT rowid rid, played_at,
            CASE WHEN played_at - LAG(played_at) OVER (ORDER BY played_at, rowid) > ?
                 THEN 1 ELSE 0 END AS new_s
          FROM play_events
        ), marks AS (
          SELECT rid, SUM(new_s) OVER (ORDER BY played_at, rowid) s
          FROM ord
        )
        UPDATE play_events SET session_id=(SELECT s FROM marks WHERE marks.rid=play_events.rowid)
        """
        .replace("?", str(SESSION_GAP)),
    )
    con.execute(
        """
        INSERT INTO listen_session
        SELECT session_id, MIN(played_at), COUNT(*) FROM play_events WHERE session_id IS NOT NULL
        GROUP BY session_id
        """
    )
    n = con.execute("SELECT COUNT(*) FROM listen_session").fetchone()[0]
    con.commit()
    return n


def affinity(con: sqlite3.Connection) -> dict:
    con.execute("DROP TABLE IF EXISTS track_affinity")
    con.execute("CREATE INDEX IF NOT EXISTS pe_session ON play_events(session_id, track_id)")
    con.executescript(
        """
        CREATE TABLE track_affinity(
          a INTEGER, b INTEGER, co INTEGER, pmi REAL,
          PRIMARY KEY(a,b)) WITHOUT ROWID;
        """
    )
    n_events = con.execute("SELECT COUNT(*) FROM play_events").fetchone()[0]
    n_tracks = con.execute("SELECT COUNT(DISTINCT track_id) FROM play_events").fetchone()[0]
    # co-play counts within sessions (dedupe repeated track in same session? keep raw counts, cap influence by sqrt later in model)
    pairs = con.execute(
        """
        SELECT e1.track_id a, e2.track_id b, COUNT(*) c
        FROM play_events e1 JOIN play_events e2
          ON e1.session_id=e2.session_id AND e1.track_id<e2.track_id
        WHERE e1.session_id IS NOT NULL
        GROUP BY a,b HAVING c>=?
        """.replace("?", str(PMI_MIN_CO))
    ).fetchall()
    plays = dict(con.execute("SELECT track_id, COUNT(*) FROM play_events GROUP BY track_id").fetchall())
    n_sess = max(1, con.execute("SELECT COUNT(*) FROM listen_session").fetchone()[0])

    # PMI with smoothing; keep topK per node via heap pass
    from collections import defaultdict

    best: dict[int, list] = defaultdict(list)
    for a, b, c in pairs:
        pa = (plays.get(a, 0) + 1) / (n_events + n_tracks)
        pb = (plays.get(b, 0) + 1) / (n_events + n_tracks)
        pab = (c + 1) / (n_sess + 1)
        pmi = math.log(pab / (pa * pb))
        for x, y, cc in ((a, b, c), (b, a, c)):
            h = best[x]
            h.append((pmi, y, cc))
    rows = []
    for x, lst in best.items():
        lst.sort(reverse=True)
        for pmi, y, c in lst[:TOPK]:
            a, b = (x, y) if x < y else (y, x)
            rows.append((a, b, c, pmi))
    seen = set()
    dedup = []
    for a, b, c, pmi in rows:
        k = (a, b)
        if k not in seen:
            seen.add(k)
            dedup.append((a, b, c, pmi))
    con.executemany("INSERT OR REPLACE INTO track_affinity VALUES(?,?,?,?)", dedup)
    con.commit()
    return {"pairs_kept": len(dup_guard := dedup), "pair_types": len(pairs)}  # noqa: F841


def main(argv=None) -> int:
    from .db import connect

    ap = argparse.ArgumentParser(prog="pgen-build-affinity")
    ap.add_argument("--db", default=str(Path.home() / ".local/share/pgen/v2.db"))
    args = ap.parse_args(argv)
    con = connect(args.db)
    n_sess = sessionize(con)
    stats = affinity(con)
    lens = con.execute(
        "SELECT AVG(n_tracks), COUNT(*) FROM listen_session WHERE n_tracks>=2"
    ).fetchone()
    print({"sessions": n_sess, "avg_tracks_per_multi_session": lens[0], "multi_sessions": lens[1]})
    print(stats)
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
