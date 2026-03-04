"""
Tests for playlistgen/playlist_builder.py
Covers: cap_artist, fill_short_pool, reorder_playlist, save_m3u, build_playlists
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(rows):
    """Build a DataFrame from a list of dicts; sets a default 'Score' if missing."""
    df = pd.DataFrame(rows)
    if "Score" not in df.columns:
        df["Score"] = 1.0
    return df


def _make_track(artist="ArtistA", name="TrackX", location="/music/track.mp3", score=1.0, **kwargs):
    """Return a single-track dict."""
    d = {"Artist": artist, "Name": name, "Location": location, "Score": score}
    d.update(kwargs)
    return d


def _make_cluster(n_tracks=10, artist="Artist", base_name="Track", location_tmpl="/music/{}.mp3"):
    """Create a simple cluster DataFrame with n unique tracks from one artist."""
    rows = [
        _make_track(
            artist=artist,
            name=f"{base_name}{i}",
            location=location_tmpl.format(i),
            score=float(n_tracks - i),
        )
        for i in range(n_tracks)
    ]
    return _make_df(rows)


# ---------------------------------------------------------------------------
# cap_artist
# ---------------------------------------------------------------------------


class TestCapArtist:
    def test_limits_tracks_per_artist(self):
        from playlistgen.playlist_builder import cap_artist

        rows = [_make_track(artist="A", name=f"T{i}") for i in range(10)]
        df = _make_df(rows)
        result = cap_artist(df, max_per_artist=3)
        assert (result["Artist"] == "A").sum() <= 3

    def test_multiple_artists_each_capped(self):
        from playlistgen.playlist_builder import cap_artist

        rows = (
            [_make_track(artist="A", name=f"T{i}") for i in range(6)]
            + [_make_track(artist="B", name=f"T{i}") for i in range(6)]
        )
        df = _make_df(rows)
        result = cap_artist(df, max_per_artist=4)
        assert (result["Artist"] == "A").sum() <= 4
        assert (result["Artist"] == "B").sum() <= 4

    def test_fewer_tracks_than_cap_unchanged(self):
        from playlistgen.playlist_builder import cap_artist

        rows = [_make_track(artist="A", name=f"T{i}") for i in range(2)]
        df = _make_df(rows)
        result = cap_artist(df, max_per_artist=5)
        assert len(result) == 2

    def test_returns_dataframe(self):
        from playlistgen.playlist_builder import cap_artist

        df = _make_df([_make_track()])
        result = cap_artist(df, max_per_artist=2)
        assert isinstance(result, pd.DataFrame)

    def test_cap_one_per_artist(self):
        from playlistgen.playlist_builder import cap_artist

        rows = [_make_track(artist="A", name=f"T{i}") for i in range(5)]
        df = _make_df(rows)
        result = cap_artist(df, max_per_artist=1)
        assert len(result) == 1

    def test_preserves_columns(self):
        from playlistgen.playlist_builder import cap_artist

        df = _make_df([_make_track(artist="A", name="T1", location="/f.mp3")])
        result = cap_artist(df, max_per_artist=2)
        assert "Artist" in result.columns
        assert "Name" in result.columns
        assert "Location" in result.columns


# ---------------------------------------------------------------------------
# fill_short_pool
# ---------------------------------------------------------------------------


class TestFillShortPool:
    def _global_df(self, n=50):
        rows = [
            _make_track(
                artist=f"Artist{i % 10}",
                name=f"GlobalTrack{i}",
                location=f"/global/{i}.mp3",
                score=float(i),
            )
            for i in range(n)
        ]
        return _make_df(rows)

    def test_backfills_to_target(self):
        from playlistgen.playlist_builder import fill_short_pool

        cluster = _make_cluster(n_tracks=5)
        global_df = self._global_df(50)
        result = fill_short_pool(cluster, global_df, target_len=20, max_per_artist=4)
        assert len(result) <= 20
        assert len(result) >= 5  # original tracks kept

    def test_no_duplicates_added(self):
        from playlistgen.playlist_builder import fill_short_pool

        cluster = _make_cluster(n_tracks=5, artist="Artist0")
        global_df = self._global_df(50)
        result = fill_short_pool(cluster, global_df, target_len=15, max_per_artist=4)
        dupes = result.duplicated(subset=["Artist", "Name"])
        assert not dupes.any()

    def test_does_not_exceed_target_len(self):
        from playlistgen.playlist_builder import fill_short_pool

        cluster = _make_cluster(n_tracks=3)
        global_df = self._global_df(100)
        result = fill_short_pool(cluster, global_df, target_len=10, max_per_artist=4)
        assert len(result) <= 10

    def test_already_at_target_not_modified(self):
        from playlistgen.playlist_builder import fill_short_pool

        cluster = _make_cluster(n_tracks=10)
        global_df = self._global_df(50)
        result = fill_short_pool(cluster, global_df, target_len=10, max_per_artist=4)
        # No backfill needed — same rows
        assert len(result) == 10

    def test_empty_global_returns_original(self):
        from playlistgen.playlist_builder import fill_short_pool

        cluster = _make_cluster(n_tracks=3)
        # Build an empty DataFrame that has the required 'Artist' and 'Name' columns
        global_df = pd.DataFrame(columns=["Artist", "Name", "Location", "Score"])
        result = fill_short_pool(cluster, global_df, target_len=10, max_per_artist=4)
        assert len(result) == 3

    def test_respects_max_per_artist(self):
        from playlistgen.playlist_builder import fill_short_pool

        # cluster already has 4 tracks from ArtistA (at the limit)
        rows = [_make_track(artist="ArtistA", name=f"T{i}") for i in range(4)]
        cluster = _make_df(rows)

        # global_df has more tracks from ArtistA and other artists
        global_rows = (
            [_make_track(artist="ArtistA", name=f"G{i}") for i in range(10)]
            + [_make_track(artist="ArtistB", name=f"B{i}") for i in range(10)]
        )
        global_df = _make_df(global_rows)

        result = fill_short_pool(cluster, global_df, target_len=10, max_per_artist=4)
        # ArtistA is already at max_per_artist=4, so no new ArtistA tracks should be added
        assert (result["Artist"] == "ArtistA").sum() == 4


# ---------------------------------------------------------------------------
# reorder_playlist
# ---------------------------------------------------------------------------


class TestReorderPlaylist:
    def test_returns_all_tracks(self):
        from playlistgen.playlist_builder import reorder_playlist

        cluster = _make_cluster(n_tracks=10)
        result = reorder_playlist(cluster)
        assert len(result) == 10

    def test_returns_dataframe(self):
        from playlistgen.playlist_builder import reorder_playlist

        cluster = _make_cluster(n_tracks=5)
        result = reorder_playlist(cluster)
        assert isinstance(result, pd.DataFrame)

    def test_no_tracks_lost(self):
        from playlistgen.playlist_builder import reorder_playlist

        rows = (
            [_make_track(artist="A", name=f"T{i}") for i in range(5)]
            + [_make_track(artist="B", name=f"T{i}") for i in range(5)]
        )
        df = _make_df(rows)
        result = reorder_playlist(df)
        assert len(result) == 10

    def test_columns_preserved(self):
        from playlistgen.playlist_builder import reorder_playlist

        df = _make_df([_make_track()])
        result = reorder_playlist(df)
        for col in ["Artist", "Name", "Location", "Score"]:
            assert col in result.columns

    def test_bpm_arc_ordering(self):
        from playlistgen.playlist_builder import reorder_playlist

        # Create 30 tracks with varied BPM so energy-arc kicks in (>=30% valid)
        rows = []
        for i in range(30):
            rows.append({
                "Artist": f"Artist{i % 5}",
                "Name": f"Track{i}",
                "Location": f"/m/{i}.mp3",
                "Score": 1.0,
                "BPM": 80 + i * 3,  # 80..167
            })
        df = _make_df(rows)
        result = reorder_playlist(df)
        assert len(result) == 30

    def test_round_robin_without_bpm(self):
        from playlistgen.playlist_builder import reorder_playlist

        rows = (
            [_make_track(artist="A", name=f"T{i}") for i in range(5)]
            + [_make_track(artist="B", name=f"T{i}") for i in range(5)]
        )
        df = _make_df(rows)  # no BPM column
        result = reorder_playlist(df)
        assert len(result) == 10


# ---------------------------------------------------------------------------
# save_m3u
# ---------------------------------------------------------------------------


_FAKE_CFG = {
    "OUTPUT_DIR": "/tmp/test_m3u_output",
    "CLUSTER_COUNT": 6,
    "MAX_PER_ARTIST": 4,
    "TRACKS_PER_MIX": 50,
}


class TestSaveM3u:
    def test_creates_m3u_file(self, tmp_path):
        from playlistgen.playlist_builder import save_m3u

        with patch("playlistgen.playlist_builder.load_config", return_value=_FAKE_CFG):
            df = _make_df([_make_track(location="/music/song.mp3")])
            out = save_m3u(df, label="Test Playlist", out_dir=str(tmp_path))
        assert out.exists()
        assert out.suffix == ".m3u"

    def test_m3u_has_extm3u_header(self, tmp_path):
        from playlistgen.playlist_builder import save_m3u

        with patch("playlistgen.playlist_builder.load_config", return_value=_FAKE_CFG):
            df = _make_df([_make_track(location="/music/song.mp3")])
            out = save_m3u(df, label="Test Playlist", out_dir=str(tmp_path))
        content = out.read_text()
        assert content.startswith("#EXTM3U")

    def test_m3u_contains_extinf(self, tmp_path):
        from playlistgen.playlist_builder import save_m3u

        with patch("playlistgen.playlist_builder.load_config", return_value=_FAKE_CFG):
            df = _make_df([_make_track(artist="ArtistA", name="SongB", location="/m/s.mp3")])
            out = save_m3u(df, label="Test", out_dir=str(tmp_path))
        content = out.read_text()
        assert "#EXTINF:" in content
        assert "ArtistA" in content
        assert "SongB" in content

    def test_m3u_contains_file_path(self, tmp_path):
        from playlistgen.playlist_builder import save_m3u

        with patch("playlistgen.playlist_builder.load_config", return_value=_FAKE_CFG):
            df = _make_df([_make_track(location="/music/my_song.mp3")])
            out = save_m3u(df, label="Test", out_dir=str(tmp_path))
        content = out.read_text()
        assert "/music/my_song.mp3" in content

    def test_skips_tracks_with_missing_location(self, tmp_path):
        from playlistgen.playlist_builder import save_m3u

        with patch("playlistgen.playlist_builder.load_config", return_value=_FAKE_CFG):
            rows = [
                _make_track(name="Good", location="/music/good.mp3"),
                _make_track(name="Bad", location=None),
                _make_track(name="Empty", location=""),
            ]
            df = _make_df(rows)
            out = save_m3u(df, label="Test", out_dir=str(tmp_path))
        content = out.read_text()
        assert "Good" in content
        assert "Bad" not in content

    def test_skips_nan_location(self, tmp_path):
        from playlistgen.playlist_builder import save_m3u
        import numpy as np

        with patch("playlistgen.playlist_builder.load_config", return_value=_FAKE_CFG):
            df = _make_df([_make_track(name="NanTrack", location=float("nan"))])
            out = save_m3u(df, label="Test", out_dir=str(tmp_path))
        content = out.read_text()
        assert "NanTrack" not in content

    def test_duration_written_when_column_present(self, tmp_path):
        from playlistgen.playlist_builder import save_m3u

        with patch("playlistgen.playlist_builder.load_config", return_value=_FAKE_CFG):
            rows = [_make_track(location="/m/s.mp3", **{"Duration": 240})]
            df = _make_df(rows)
            out = save_m3u(df, label="Test", out_dir=str(tmp_path))
        content = out.read_text()
        assert "#EXTINF:240," in content

    def test_duration_minus_one_when_column_missing(self, tmp_path):
        from playlistgen.playlist_builder import save_m3u

        with patch("playlistgen.playlist_builder.load_config", return_value=_FAKE_CFG):
            df = _make_df([_make_track(location="/m/s.mp3")])  # no Duration
            out = save_m3u(df, label="Test", out_dir=str(tmp_path))
        content = out.read_text()
        assert "#EXTINF:-1," in content

    def test_file_url_decoded(self, tmp_path):
        from playlistgen.playlist_builder import save_m3u

        with patch("playlistgen.playlist_builder.load_config", return_value=_FAKE_CFG):
            df = _make_df([_make_track(location="file://localhost/music/my%20song.mp3")])
            out = save_m3u(df, label="Test", out_dir=str(tmp_path))
        content = out.read_text()
        # URL encoding should be decoded; space encoded as %20
        assert "%20" not in content
        assert "my song.mp3" in content

    def test_label_sanitized_in_filename(self, tmp_path):
        from playlistgen.playlist_builder import save_m3u

        with patch("playlistgen.playlist_builder.load_config", return_value=_FAKE_CFG):
            df = _make_df([_make_track(location="/m/s.mp3")])
            out = save_m3u(df, label="Rock/Pop", out_dir=str(tmp_path))
        # The slash in label should be sanitized in the filename
        assert "/" not in out.name

    def test_creates_output_dir_if_missing(self, tmp_path):
        from playlistgen.playlist_builder import save_m3u

        new_dir = tmp_path / "new" / "subdir"
        with patch("playlistgen.playlist_builder.load_config", return_value=_FAKE_CFG):
            df = _make_df([_make_track(location="/m/s.mp3")])
            out = save_m3u(df, label="Test", out_dir=str(new_dir))
        assert new_dir.exists()
        assert out.exists()

    def test_multiple_tracks_written(self, tmp_path):
        from playlistgen.playlist_builder import save_m3u

        with patch("playlistgen.playlist_builder.load_config", return_value=_FAKE_CFG):
            rows = [_make_track(name=f"Track{i}", location=f"/m/{i}.mp3") for i in range(5)]
            df = _make_df(rows)
            out = save_m3u(df, label="Test", out_dir=str(tmp_path))
        content = out.read_text()
        # Should have 5 EXTINF lines + 5 path lines + 1 header
        extinf_count = content.count("#EXTINF:")
        assert extinf_count == 5


# ---------------------------------------------------------------------------
# build_playlists
# ---------------------------------------------------------------------------


class TestBuildPlaylists:
    def _config(self):
        return {
            "TRACKS_PER_MIX": 10,
            "MAX_PER_ARTIST": 4,
            "NUM_PLAYLISTS": 99,
            "OUTPUT_DIR": "/tmp/bp_test_output",
        }

    def test_returns_list_of_tuples(self, tmp_path):
        from playlistgen.playlist_builder import build_playlists

        cluster = _make_cluster(n_tracks=15)
        global_df = _make_cluster(n_tracks=30, artist="OtherArtist")

        with patch("playlistgen.playlist_builder.load_config", return_value=self._config()), \
             patch("playlistgen.playlist_builder.save_m3u") as mock_save:
            mock_save.return_value = tmp_path / "fake.m3u"
            result = build_playlists(
                clusters=[cluster],
                global_df=global_df,
                tracks_per_mix=10,
                max_per_artist=4,
                save=True,
                out_dir=str(tmp_path),
            )

        assert isinstance(result, list)
        assert len(result) == 1
        label, df = result[0]
        assert isinstance(label, str)
        assert isinstance(df, pd.DataFrame)

    def test_default_label_format(self, tmp_path):
        from playlistgen.playlist_builder import build_playlists

        cluster = _make_cluster(n_tracks=5)
        global_df = _make_cluster(n_tracks=30, artist="Other")

        with patch("playlistgen.playlist_builder.load_config", return_value=self._config()), \
             patch("playlistgen.playlist_builder.save_m3u") as mock_save:
            mock_save.return_value = tmp_path / "fake.m3u"
            result = build_playlists(
                [cluster], global_df, tracks_per_mix=5, max_per_artist=4,
                save=True, out_dir=str(tmp_path),
            )

        label, _ = result[0]
        assert "Cluster 1" == label

    def test_custom_name_fn(self, tmp_path):
        from playlistgen.playlist_builder import build_playlists

        cluster = _make_cluster(n_tracks=5)
        global_df = _make_cluster(n_tracks=30, artist="Other")

        with patch("playlistgen.playlist_builder.load_config", return_value=self._config()), \
             patch("playlistgen.playlist_builder.save_m3u") as mock_save:
            mock_save.return_value = tmp_path / "fake.m3u"
            result = build_playlists(
                [cluster], global_df, tracks_per_mix=5, max_per_artist=4,
                save=True, name_fn=lambda df, i: f"Custom {i}",
                out_dir=str(tmp_path),
            )

        label, _ = result[0]
        assert label == "Custom 0"

    def test_save_false_does_not_call_save_m3u(self, tmp_path):
        from playlistgen.playlist_builder import build_playlists

        cluster = _make_cluster(n_tracks=5)
        global_df = _make_cluster(n_tracks=30, artist="Other")

        with patch("playlistgen.playlist_builder.load_config", return_value=self._config()), \
             patch("playlistgen.playlist_builder.save_m3u") as mock_save:
            build_playlists(
                [cluster], global_df, tracks_per_mix=5, max_per_artist=4,
                save=False, out_dir=str(tmp_path),
            )
        mock_save.assert_not_called()

    def test_num_playlists_cap(self, tmp_path):
        from playlistgen.playlist_builder import build_playlists

        clusters = [_make_cluster(n_tracks=5, artist=f"Artist{i}") for i in range(5)]
        global_df = _make_cluster(n_tracks=50, artist="Global")

        with patch("playlistgen.playlist_builder.load_config", return_value=self._config()), \
             patch("playlistgen.playlist_builder.save_m3u") as mock_save:
            mock_save.return_value = tmp_path / "fake.m3u"
            result = build_playlists(
                clusters, global_df, tracks_per_mix=5, max_per_artist=4,
                num_playlists=2, save=True, out_dir=str(tmp_path),
            )

        assert len(result) == 2

    def test_playlist_length_capped_at_tracks_per_mix(self, tmp_path):
        from playlistgen.playlist_builder import build_playlists

        # Cluster with many tracks
        cluster = _make_cluster(n_tracks=30)
        global_df = _make_cluster(n_tracks=50, artist="Other")

        with patch("playlistgen.playlist_builder.load_config", return_value=self._config()), \
             patch("playlistgen.playlist_builder.save_m3u") as mock_save:
            mock_save.return_value = tmp_path / "fake.m3u"
            result = build_playlists(
                [cluster], global_df, tracks_per_mix=10, max_per_artist=4,
                save=True, out_dir=str(tmp_path),
            )

        _, playlist_df = result[0]
        assert len(playlist_df) <= 10

    def test_multiple_clusters_produce_multiple_playlists(self, tmp_path):
        from playlistgen.playlist_builder import build_playlists

        clusters = [_make_cluster(n_tracks=5, artist=f"Art{i}") for i in range(3)]
        global_df = _make_cluster(n_tracks=30, artist="Global")

        with patch("playlistgen.playlist_builder.load_config", return_value=self._config()), \
             patch("playlistgen.playlist_builder.save_m3u") as mock_save:
            mock_save.return_value = tmp_path / "fake.m3u"
            result = build_playlists(
                clusters, global_df, tracks_per_mix=5, max_per_artist=4,
                save=True, out_dir=str(tmp_path),
            )

        assert len(result) == 3

    def test_save_true_calls_save_m3u_per_cluster(self, tmp_path):
        from playlistgen.playlist_builder import build_playlists

        clusters = [_make_cluster(n_tracks=5, artist=f"Art{i}") for i in range(3)]
        global_df = _make_cluster(n_tracks=30, artist="Global")

        with patch("playlistgen.playlist_builder.load_config", return_value=self._config()), \
             patch("playlistgen.playlist_builder.save_m3u") as mock_save:
            mock_save.return_value = tmp_path / "fake.m3u"
            build_playlists(
                clusters, global_df, tracks_per_mix=5, max_per_artist=4,
                save=True, out_dir=str(tmp_path),
            )

        assert mock_save.call_count == 3
