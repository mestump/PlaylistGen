"""M3 stage 1: sessionize listen events and materialize track co-play affinity.

Pure SQL / disk-backed temp (OOM lesson 2026-10-04: the python-side pair
materialization hit 20GB RSS and took the box down; sqlite does it in ~100MB).

Session: consecutive plays >30min apart start a new session (global timeline).
Affinity: PMI over session co-play, +1 smoothed, top-200 partners per track,
stored once per unordered pair.

Tables:
  listen_session(session_id, started, n_tracks); play_events.session_id backfill
  track_affinity(a, b, co, pmi) with a<b
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

SESSION_GAP = 1800  # seconds
PMI_MIN_CO = 2
TOPK = 200


def sessionize(con: sqlite3.Connection) -> int:
    con.execute("DROP TABLE IF EXISTS listen_session")
    con.execute(
        "CREATE TABLE listen_session("
        "session_id INTEGER PRIMARY KEY, started INTEGER, n_tracks INTEGER)"
    )
    con.execute("UPDATE play_events SET session_id=NULL")
    # session id = cumsum of gap-start flags over global time order; sid temp
    # table is index-joined back (indexed correlated lookup, NOT the O(n^2)
    # unindexed-correlated UPDATE that stalled the first attempt)
    con.execute(
        f"""
        CREATE TEMP TABLE sid AS
        SELECT rowid AS rid, SUM(new_s) OVER (ORDER BY played_at, rowid) AS s
        FROM (
          SELECT rowid, played_at,
            CASE WHEN played_at - LAG(played_at) OVER (ORDER BY played_at, rowid) > {SESSION_GAP}
                 THEN 1 ELSE 0 END AS new_s
          FROM play_events
        )
        """
    )
    con.execute("CREATE UNIQUE INDEX temp.sid_rid ON sid(rid)")
    con.execute("UPDATE play_events SET session_id=(SELECT s FROM temp.sid WHERE sid.rid=play_events.rowid)")
    con.execute(
        "INSERT INTO listen_session "
        "SELECT session_id, MIN(played_at), COUNT(*) FROM play_events "
        "WHERE session_id IS NOT NULL GROUP BY session_id"
    )
    n = con.execute("SELECT COUNT(*) FROM listen_session").fetchone()[0]
    con.commit()
    return n


def affinity(con: sqlite3.Connection) -> dict:
    con.execute("PRAGMA cache_size=-65536")  # 64MiB page cache
    con.execute("PRAGMA temp_store=FILE")
    con.execute("CREATE INDEX IF NOT EXISTS pe_session ON play_events(session_id, track_id)")
    con.execute("DROP TABLE IF EXISTS track_affinity")
    # co-play counts: materialized temp (disk-backed), never pulled into python
    con.execute(
        f"""
        CREATE TEMP TABLE pair_co AS
        SELECT e1.track_id AS a, e2.track_id AS b, COUNT(*) AS c
        FROM play_events e1
        JOIN play_events e2 ON e2.session_id=e1.session_id AND e2.track_id>e1.track_id
        WHERE e1.session_id IS NOT NULL
        GROUP BY 1,2
        HAVING COUNT(*)>={PMI_MIN_CO}
        """
    )
    con.execute(
        f"""
        CREATE TABLE track_affinity AS
        WITH plays AS (SELECT track_id, COUNT(*) n FROM play_events GROUP BY 1),
        tot AS (
          SELECT (SELECT COUNT(*) FROM play_events) AS ne,
                 (SELECT COUNT(*) FROM listen_session) AS ns,
                 (SELECT COUNT(*) FROM tracks) AS nt
        ),
        sym AS (SELECT a,b,c FROM pair_co UNION ALL SELECT b,a,c FROM pair_co),
        scored AS (
          SELECT s.a, s.b, s.c,
            ln((s.c+1.0)/(t.ns+1.0))
            - ln((COALESCE(pa.n,0)+1.0)/(t.ne+t.nt))
            - ln((COALESCE(pb.n,0)+1.0)/(t.ne+t.nt)) AS pmi
          FROM sym s CROSS JOIN tot t
          LEFT JOIN plays pa ON pa.track_id=s.a
          LEFT JOIN plays pb ON pb.track_id=s.b
        ),
        ranked AS (
          SELECT a,b,c,pmi, ROW_NUMBER() OVER (PARTITION BY a ORDER BY pmi DESC, c DESC) rk
          FROM scored
        )
        SELECT MIN(a) AS a, MAX(b) AS b,
               MAX(c) AS c, MAX(pmi) AS pmi
        FROM ranked WHERE rk<={TOPK}
        GROUP BY CASE WHEN a<b THEN a*100000000+b ELSE b*100000000+a END
        """
    )
    con.execute("CREATE INDEX IF NOT EXISTS aff_a ON track_affinity(a)")
    con.execute("CREATE INDEX IF NOT EXISTS aff_b ON track_affinity(b)")
    n_pairs = con.execute("SELECT COUNT(*) FROM track_affinity").fetchone()[0]
    con.commit()
    return {"pairs": n_pairs}


def main(argv=None) -> int:
    from .db import connect

    ap = argparse.ArgumentParser(prog="pgen-build-affinity")
    ap.add_argument("--db", default=str(Path.home() / ".local/share/pgen/v2.db"))
    args = ap.parse_args(argv)
    con = connect(args.db)
    n_sess = sessionize(con)
    multi = con.execute(
        "SELECT COUNT(*), AVG(n_tracks) FROM listen_session WHERE n_tracks>=2"
    ).fetchone()
    stats = affinity(con)
    print({"sessions": n_sess, "multi_sessions": multi[0], "avg_tracks_multi": round(multi[1] or 0, 2)})
    print(stats)
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
