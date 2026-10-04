"""M2 stage 2: materialize per-track behavioral signals into track_signal.

Pure SQL over play_events/source_counts/tracks. Rebuildable: --rebuild drops first.
Signals:
  plays_90d, plays_365d, plays_events_total (timestamped, all sources)
  plays_total_all (events + counter sources, deduped by source priority)
  skip_ratio (counter sources, blended), days_since_play, artist_plays_365d,
  artist_rank (by artist_plays_365d), hour histogram peak bucket

Source priority for totals: spotify events are truth-per-play; counters (ipod/
strawberry/itunes_legacy) are aggregates -- they are NOT added to event counts
(except when a track has no events at all, then counters stand alone).
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path


def rebuild(con: sqlite3.Connection) -> dict:
    now = int(time.time())
    con.execute("DROP TABLE IF EXISTS track_signal")
    con.executescript(
        """
        CREATE TABLE track_signal(
          track_id INTEGER PRIMARY KEY REFERENCES tracks(id) ON DELETE CASCADE,
          plays_90d REAL, plays_365d REAL, plays_events INTEGER,
          plays_counters INTEGER, plays_total REAL,
          skips INTEGER, skip_ratio REAL,
          last_played_ts INTEGER, days_since_play REAL,
          artist_key TEXT, artist_plays_365d REAL, artist_rank INTEGER,
          hour_peak INTEGER
        );
        """
    )
    con.execute(
        """
        INSERT INTO track_signal
        (track_id, plays_90d, plays_365d, plays_events, plays_counters, plays_total,
         skips, skip_ratio, last_played_ts, days_since_play)
        WITH ev AS (
          SELECT track_id,
                 SUM(CASE WHEN played_at > ? - 90*86400 THEN 1 ELSE 0 END) p90,
                 SUM(CASE WHEN played_at > ? - 365*86400 THEN 1 ELSE 0 END) p365,
                 COUNT(*) pall, MAX(played_at) lts
          FROM play_events GROUP BY track_id
        ),
        ct AS (
          SELECT track_id, SUM(plays) apl, SUM(skips) ask FROM source_counts GROUP BY track_id
        )
        SELECT t.id,
               COALESCE(ev.p90, 0), COALESCE(ev.p365, 0), COALESCE(ev.pall, 0),
               COALESCE(ct.apl, 0),
               CASE WHEN ev.pall IS NOT NULL THEN ev.pall ELSE COALESCE(ct.apl, 0) END,
               COALESCE(ct.ask, 0),
               CASE WHEN COALESCE(ct.apl, 0) > 0 THEN CAST(ct.ask AS REAL)/ct.apl END,
               COALESCE(ev.lts, sc.lastplayed),
               CASE WHEN COALESCE(ev.lts, sc.lastplayed) IS NOT NULL
                    THEN (? - COALESCE(ev.lts, sc.lastplayed)) / 86400.0 END
        FROM tracks t
        LEFT JOIN ev ON ev.track_id = t.id
        LEFT JOIN ct ON ct.track_id = t.id
        LEFT JOIN source_counts sc ON sc.track_id = t.id AND sc.source =
             (SELECT source FROM source_counts s2 WHERE s2.track_id=t.id
              ORDER BY lastplayed DESC NULLS LAST LIMIT 1)
        """,
        (now, now, now),
    )
    # artist rollups on 365d window + artist rank
    con.execute("UPDATE track_signal SET artist_key=(SELECT artist_canon FROM tracks WHERE id=track_id)")
    con.execute(
        """
        UPDATE track_signal SET artist_plays_365d=(
          SELECT SUM(plays_365d) FROM track_signal s2 WHERE s2.artist_key=track_signal.artist_key
        )
        """
    )
    con.execute(
        """
        UPDATE track_signal SET artist_rank=(
          SELECT rk FROM (
            SELECT artist_key, ROW_NUMBER() OVER (ORDER BY SUM(plays_365d) DESC) rk
            FROM track_signal GROUP BY artist_key
          ) r WHERE r.artist_key = track_signal.artist_key
        )
        """
    )
    # hour-of-day peak from events (utc-4 approx; good enough for taste buckets)
    con.execute(
        """
        UPDATE track_signal SET hour_peak=(
          SELECT hour FROM (
            SELECT ((played_at % 86400)/3600 - 4 + 24) % 24 AS hour, COUNT(*) n
            FROM play_events WHERE track_id=track_signal.track_id GROUP BY hour
            ORDER BY n DESC LIMIT 1)
        ) WHERE hour_peak IS NULL
        """
    )
    con.commit()
    n = con.execute("SELECT COUNT(*) FROM track_signal").fetchone()[0]
    return {"signal_rows": n}


def summary(con: sqlite3.Connection) -> dict:
    q = lambda s: con.execute(s).fetchone()[0]  # noqa: E731
    return {
        "tracks_with_any_signal": q("SELECT COUNT(*) FROM track_signal WHERE plays_total>0 OR plays_events>0"),
        "events_90d": q("SELECT COALESCE(SUM(plays_90d),0) FROM track_signal"),
        "events_365d": q("SELECT COALESCE(SUM(plays_365d),0) FROM track_signal"),
        "never_played": q("SELECT COUNT(*) FROM track_signal WHERE COALESCE(plays_total,0)=0"),
        "with_skipdata": q("SELECT COUNT(*) FROM track_signal WHERE skips>0"),
        "artists": q("SELECT COUNT(DISTINCT artist_key) FROM track_signal WHERE artist_key IS NOT NULL"),
    }


def main(argv=None) -> int:
    from .db import connect

    ap = argparse.ArgumentParser(prog="pgen-build-signals")
    ap.add_argument("--db", default=str(Path.home() / ".local/share/pgen/v2.db"))
    args = ap.parse_args(argv)
    con = connect(args.db)
    stats = rebuild(con)
    print(stats)
    print(summary(con))
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
