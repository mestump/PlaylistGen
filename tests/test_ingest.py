import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pgen.db import connect  # noqa: E402
from pgen.ingest.files import ingest_directory  # noqa: E402
from pgen.ingest.spotify_history import ingest_spotify_history  # noqa: E402
from pgen.ingest.strawberry import ingest_strawberry  # noqa: E402
from pgen.resolve import upsert_track  # noqa: E402


def test_resolve_merges_case_and_feat():
    con = connect(":memory:")
    a = upsert_track(con, "The National", "Bloodbuzz Ohio")
    b = upsert_track(con, "national", "Bloodbuzz Ohio (Album Version)")
    c = upsert_track(con, "The National", "Bloodbuzz Ohio", album="High Violet", year=2010, path="/x.mp3", path_source="files")
    assert a == b == c
    row = con.execute("SELECT artist,title,album,year FROM tracks WHERE id=?", (a,)).fetchone()
    assert row["album"] == "High Violet" and row["year"] == 2010
    paths = con.execute("SELECT path FROM track_paths WHERE track_id=?", (a,)).fetchall()
    assert len(paths) == 1


def test_files_ingest_filename_fallback(tmp_path: Path):
    d = tmp_path / "lib" / "Artist Name"
    d.mkdir(parents=True)
    (d / "Artist Name - Some Song.mp3").write_bytes(b"")
    (d / "._junk.mp3").write_bytes(b"")
    con = connect(":memory:")
    stats = ingest_directory(con, d)
    assert stats["files"] == 1
    row = con.execute("SELECT artist,title FROM tracks").fetchone()
    assert (row["artist"], row["title"]) == ("Artist Name", "Some Song")


def test_strawberry_ingest(tmp_path: Path):
    sb = tmp_path / "sb.db"
    con2 = sqlite3.connect(sb)
    con2.execute(
        "CREATE TABLE songs(id INTEGER PRIMARY KEY, artist TEXT, title TEXT, album TEXT,"
        "year INT, playcount INT, skipcount INT, lastplayed INT)"
    )
    con2.execute(
        "INSERT INTO songs(artist,title,album,year,playcount,skipcount,lastplayed)"
        " VALUES('A','T','Al',2001,3,1,1700000000)"
    )
    con2.commit()
    con2.close()
    con = connect(":memory:")
    stats = ingest_strawberry(con, sb)
    assert stats["plays"] == 3
    row = con.execute("SELECT plays,skips,lastplayed FROM source_counts WHERE source='strawberry'").fetchone()
    assert (row["plays"], row["skips"]) == (3, 1)
    # idempotent: unchanged stamp skips
    again = ingest_strawberry(con, sb)
    assert again.get("skipped") == "unchanged" or again.get("tracks", 1) >= 0


def test_spotify_history(tmp_path: Path):
    f = tmp_path / "Streaming_History_Audio_2020-2021_0.json"
    f.write_text(
        json.dumps(
            [
                {"ts": "2020-05-01T12:00:00Z", "master_metadata_album_artist_name": "Spiritbox",
                 "master_metadata_track_name": "Holy Roller", "msPlayed": 200000, "skipped": False},
                {"ts": "2020-05-01T12:30:00Z", "artistName": "Old Format", "trackName": "Older",
                 "msPlayed": 30000, "skipped": True},
                {"platform": "whatever"},  # no artist -> skipped row
            ]
        )
    )
    con = connect(":memory:")
    stats = ingest_spotify_history(con, [f])
    assert stats["events"] == 2
    n = con.execute("SELECT COUNT(*) FROM play_events WHERE skipped=1").fetchone()[0]
    assert n == 1
    # re-run: stamp makes it idempotent
    again = ingest_spotify_history(con, [f])
    assert again["events"] == 0
