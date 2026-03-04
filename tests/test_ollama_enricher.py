"""
Tests for playlistgen/enrichers/ollama_enricher.py

All HTTP traffic is intercepted with unittest.mock so no real Ollama instance
is required.
"""

import json
import sqlite3
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------

def _import_module():
    """Import the module under test, reloading so module-level flags reset."""
    import importlib
    import playlistgen.enrichers.ollama_enricher as mod
    importlib.reload(mod)
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_df():
    """Three-track DataFrame ready for enrichment (no Mood/Energy/Valence)."""
    return pd.DataFrame(
        {
            "Artist": ["Radiohead", "Massive Attack", "Portishead"],
            "Name": ["Creep", "Teardrop", "Glory Box"],
            "Genre": ["Alternative", "Trip-Hop", "Trip-Hop"],
            "BPM": [92.0, 80.0, 88.0],
        }
    )


def _make_ollama_response(items: list) -> MagicMock:
    """Return a mock requests.Response representing a successful Ollama reply."""
    resp = MagicMock()
    resp.ok = True
    resp.status_code = 200
    content = json.dumps(items)
    resp.json.return_value = {
        "choices": [{"message": {"content": content}}]
    }
    return resp


def _enrichment_items(n=3):
    """Return n realistic enrichment dicts (1-indexed)."""
    moods = ["Sad", "Chill", "Dreamy"]
    return [
        {
            "idx": i + 1,
            "mood": moods[i % len(moods)],
            "energy": 4 + i,
            "valence": 3 + i,
            "tags": ["atmospheric", "dark", "melancholy"],
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# _parse_json_response
# ---------------------------------------------------------------------------

class TestParseJsonResponse:
    def _fn(self):
        return _import_module()._parse_json_response

    def test_handles_clean_json_array(self):
        parse = self._fn()
        data = [{"idx": 1, "mood": "Happy", "energy": 7, "valence": 8}]
        result = parse(json.dumps(data))
        assert result == data

    def test_handles_markdown_json_code_fence(self):
        parse = self._fn()
        data = [{"idx": 1, "mood": "Chill", "energy": 4, "valence": 6}]
        raw = "```json\n" + json.dumps(data) + "\n```"
        result = parse(raw)
        assert result is not None
        assert len(result) == 1
        assert result[0]["mood"] == "Chill"

    def test_handles_markdown_plain_code_fence(self):
        """Fences without a language tag must also be stripped."""
        parse = self._fn()
        data = [{"idx": 1, "mood": "Energetic", "energy": 9, "valence": 7}]
        raw = "```\n" + json.dumps(data) + "\n```"
        result = parse(raw)
        assert result is not None
        assert result[0]["mood"] == "Energetic"

    def test_returns_none_for_invalid_json(self):
        parse = self._fn()
        result = parse("this is not JSON at all")
        assert result is None

    def test_returns_none_for_empty_string(self):
        parse = self._fn()
        result = parse("")
        assert result is None

    def test_returns_none_for_json_object_not_array(self):
        """The function must only accept a top-level JSON array."""
        parse = self._fn()
        result = parse(json.dumps({"mood": "Happy"}))
        assert result is None

    def test_extracts_array_embedded_in_prose(self):
        """If a JSON array is embedded in surrounding text it should be found."""
        parse = self._fn()
        data = [{"idx": 1, "mood": "Groovy", "energy": 8, "valence": 7}]
        raw = "Here is my answer: " + json.dumps(data) + " — done."
        result = parse(raw)
        assert result is not None
        assert result[0]["mood"] == "Groovy"

    def test_multiple_tracks_returned_intact(self):
        parse = self._fn()
        data = _enrichment_items(3)
        result = parse(json.dumps(data))
        assert result is not None
        assert len(result) == 3
        assert result[1]["mood"] == "Chill"


# ---------------------------------------------------------------------------
# batch_enrich_ollama — happy path
# ---------------------------------------------------------------------------

class TestBatchEnrichOllama:
    def test_fills_mood_energy_valence_columns(self, tmp_path, sample_df):
        mod = _import_module()
        cache_db = str(tmp_path / "test_enrich.sqlite")
        items = _enrichment_items(3)
        mock_resp = _make_ollama_response(items)

        with patch.object(mod.requests, "post", return_value=mock_resp):
            result = mod.batch_enrich_ollama(
                sample_df,
                base_url="http://localhost:11434",
                model="llama3",
                cache_db=cache_db,
            )

        assert "Mood" in result.columns
        assert "Energy" in result.columns
        assert "Valence" in result.columns

        # All three rows must have been filled
        assert result["Mood"].notna().all()
        assert result["Energy"].notna().all()
        assert result["Valence"].notna().all()

        assert list(result["Mood"]) == ["Sad", "Chill", "Dreamy"]
        assert list(result["Energy"]) == [4, 5, 6]
        assert list(result["Valence"]) == [3, 4, 5]

    def test_does_not_mutate_original_dataframe(self, tmp_path, sample_df):
        mod = _import_module()
        cache_db = str(tmp_path / "test_enrich.sqlite")
        items = _enrichment_items(3)
        mock_resp = _make_ollama_response(items)
        original_cols = list(sample_df.columns)

        with patch.object(mod.requests, "post", return_value=mock_resp):
            mod.batch_enrich_ollama(sample_df, cache_db=cache_db)

        # Original must be untouched
        assert list(sample_df.columns) == original_cols
        assert "Mood" not in sample_df.columns

    def test_uses_sqlite_cache_on_second_run(self, tmp_path, sample_df):
        """Second call must read from cache and issue no HTTP requests."""
        mod = _import_module()
        cache_db = str(tmp_path / "cache_test.sqlite")
        items = _enrichment_items(3)
        mock_resp = _make_ollama_response(items)

        # First run — populates cache
        with patch.object(mod.requests, "post", return_value=mock_resp) as post1:
            mod.batch_enrich_ollama(sample_df, cache_db=cache_db)
        assert post1.call_count == 1

        # Second run — should use cache, no HTTP call
        with patch.object(mod.requests, "post", return_value=mock_resp) as post2:
            result2 = mod.batch_enrich_ollama(sample_df, cache_db=cache_db)
        assert post2.call_count == 0

        # Results should still be populated from cache
        assert list(result2["Mood"]) == ["Sad", "Chill", "Dreamy"]

    def test_skips_already_enriched_tracks(self, tmp_path):
        """Rows whose Mood column is already filled must not be sent to Ollama."""
        mod = _import_module()
        cache_db = str(tmp_path / "skip_test.sqlite")

        df = pd.DataFrame(
            {
                "Artist": ["Artist1", "Artist2"],
                "Name": ["Song1", "Song2"],
                "Mood": ["Happy", None],  # Artist1 already enriched
                "Energy": [7, None],
                "Valence": [8, None],
            }
        )
        items = [{"idx": 1, "mood": "Chill", "energy": 4, "valence": 5, "tags": []}]
        mock_resp = _make_ollama_response(items)

        with patch.object(mod.requests, "post", return_value=mock_resp) as mock_post:
            result = mod.batch_enrich_ollama(df, cache_db=cache_db)

        # Only one track should be sent
        assert mock_post.call_count == 1
        user_msg = mock_post.call_args[1]["json"]["messages"][1]["content"]
        assert "Artist2" in user_msg
        assert "Artist1" not in user_msg

        # Pre-existing value must be preserved
        assert result.at[0, "Mood"] == "Happy"
        assert result.at[1, "Mood"] == "Chill"

    def test_skips_unknown_mood_values(self, tmp_path):
        """Rows with Mood='Unknown' must be treated as needing enrichment."""
        mod = _import_module()
        cache_db = str(tmp_path / "unknown_test.sqlite")

        df = pd.DataFrame(
            {
                "Artist": ["Artist1"],
                "Name": ["Song1"],
                "Mood": ["Unknown"],
            }
        )
        items = [{"idx": 1, "mood": "Happy", "energy": 7, "valence": 8, "tags": []}]
        mock_resp = _make_ollama_response(items)

        with patch.object(mod.requests, "post", return_value=mock_resp):
            result = mod.batch_enrich_ollama(df, cache_db=cache_db)

        assert result.at[0, "Mood"] == "Happy"


# ---------------------------------------------------------------------------
# batch_enrich_ollama — API error / retry behaviour
# ---------------------------------------------------------------------------

class TestBatchEnrichErrors:
    def test_retries_on_http_error_then_skips(self, tmp_path, sample_df):
        """On HTTP error the enricher must retry up to 3 times then skip the batch."""
        mod = _import_module()
        cache_db = str(tmp_path / "retry_test.sqlite")

        bad_resp = MagicMock()
        bad_resp.ok = False
        bad_resp.status_code = 500
        bad_resp.text = "Internal Server Error"

        with patch.object(mod.requests, "post", return_value=bad_resp) as mock_post:
            with patch("time.sleep"):  # skip actual waiting
                result = mod.batch_enrich_ollama(sample_df, cache_db=cache_db)

        # 3 attempts per batch, 1 batch for 3 tracks with default batch_size=50
        assert mock_post.call_count == 3

        # Columns exist but no values were written
        assert result["Mood"].isna().all()

    def test_retries_on_connection_error_then_skips(self, tmp_path, sample_df):
        """On network exception the enricher must retry 3 times then skip."""
        mod = _import_module()
        cache_db = str(tmp_path / "conn_err.sqlite")

        with patch.object(
            mod.requests, "post", side_effect=ConnectionError("refused")
        ) as mock_post:
            with patch("time.sleep"):
                result = mod.batch_enrich_ollama(sample_df, cache_db=cache_db)

        assert mock_post.call_count == 3
        assert result["Mood"].isna().all()

    def test_retries_on_unparseable_response(self, tmp_path, sample_df):
        """When Ollama returns malformed JSON all 3 retry attempts are exhausted."""
        mod = _import_module()
        cache_db = str(tmp_path / "parse_err.sqlite")

        bad_resp = MagicMock()
        bad_resp.ok = True
        bad_resp.status_code = 200
        bad_resp.json.return_value = {
            "choices": [{"message": {"content": "not JSON at all"}}]
        }

        with patch.object(mod.requests, "post", return_value=bad_resp) as mock_post:
            with patch("time.sleep"):
                result = mod.batch_enrich_ollama(sample_df, cache_db=cache_db)

        assert mock_post.call_count == 3
        assert result["Mood"].isna().all()

    def test_partial_batch_failure_skips_bad_batch(self, tmp_path):
        """
        If batch 1 fails but batch 2 succeeds, only the second batch's tracks
        get enriched.
        """
        mod = _import_module()
        cache_db = str(tmp_path / "partial.sqlite")

        df = pd.DataFrame(
            {
                "Artist": [f"Artist{i}" for i in range(4)],
                "Name": [f"Song{i}" for i in range(4)],
            }
        )

        # Items for the second batch (batch_size=2 → two batches)
        batch2_items = [
            {"idx": 1, "mood": "Happy", "energy": 7, "valence": 8, "tags": []},
            {"idx": 2, "mood": "Chill", "energy": 4, "valence": 5, "tags": []},
        ]

        bad_resp = MagicMock(ok=False, status_code=503, text="unavailable")
        good_resp = _make_ollama_response(batch2_items)

        # First batch: 3 consecutive failures; second batch: success on first try
        responses = [bad_resp, bad_resp, bad_resp, good_resp]

        with patch.object(mod.requests, "post", side_effect=responses):
            with patch("time.sleep"):
                result = mod.batch_enrich_ollama(
                    df, cache_db=cache_db, batch_size=2
                )

        # First two tracks (batch 1) skipped; last two (batch 2) enriched
        assert pd.isna(result.at[0, "Mood"])
        assert pd.isna(result.at[1, "Mood"])
        assert result.at[2, "Mood"] == "Happy"
        assert result.at[3, "Mood"] == "Chill"


# ---------------------------------------------------------------------------
# batch_enrich_ollama — no requests module
# ---------------------------------------------------------------------------

class TestBatchEnrichNoRequests:
    def test_returns_df_unchanged_when_requests_missing(self, tmp_path, sample_df):
        """When requests is not installed the DataFrame must be returned as-is."""
        import importlib
        import playlistgen.enrichers.ollama_enricher as mod

        original_flag = mod.REQUESTS_AVAILABLE
        try:
            mod.REQUESTS_AVAILABLE = False
            result = mod.batch_enrich_ollama(
                sample_df, cache_db=str(tmp_path / "noop.sqlite")
            )
        finally:
            mod.REQUESTS_AVAILABLE = original_flag

        # Should be a copy with the same values — no Mood/Energy/Valence added
        pd.testing.assert_frame_equal(result, sample_df)


# ---------------------------------------------------------------------------
# batch_enrich_ollama — edge cases
# ---------------------------------------------------------------------------

class TestBatchEnrichEdgeCases:
    def test_empty_dataframe_returns_immediately(self, tmp_path):
        mod = _import_module()
        cache_db = str(tmp_path / "empty.sqlite")
        df = pd.DataFrame(columns=["Artist", "Name"])

        with patch.object(mod.requests, "post") as mock_post:
            result = mod.batch_enrich_ollama(df, cache_db=cache_db)

        mock_post.assert_not_called()
        assert isinstance(result, pd.DataFrame)

    def test_tracks_missing_artist_or_name_skipped(self, tmp_path):
        """Rows with a blank artist OR a blank name must be skipped entirely.

        Note: Python's ``str(row.get("Artist") or "")`` converts NaN to "nan"
        (because float NaN is truthy), so only genuine empty strings reliably
        produce an empty result.  This test therefore uses empty strings rather
        than None to exercise the guard clause.
        """
        mod = _import_module()
        cache_db = str(tmp_path / "missing_fields.sqlite")

        df = pd.DataFrame(
            {
                # Row 0: empty artist  -> skipped
                # Row 1: empty name   -> skipped
                # Row 2: both present -> would be sent (but let's keep it absent
                #                        so we can assert no request at all)
                "Artist": ["", "ValidArtist"],
                "Name": ["Song1", ""],
            }
        )

        with patch.object(mod.requests, "post") as mock_post:
            result = mod.batch_enrich_ollama(df, cache_db=cache_db)

        # Neither row is complete, so no HTTP request should be issued
        mock_post.assert_not_called()

    def test_mood_energy_valence_columns_created_when_absent(self, tmp_path):
        """The function adds the three columns even if the input df lacks them."""
        mod = _import_module()
        cache_db = str(tmp_path / "cols_test.sqlite")

        df = pd.DataFrame({"Artist": ["A"], "Name": ["S"]})
        items = [{"idx": 1, "mood": "Groovy", "energy": 8, "valence": 7, "tags": []}]
        mock_resp = _make_ollama_response(items)

        with patch.object(mod.requests, "post", return_value=mock_resp):
            result = mod.batch_enrich_ollama(df, cache_db=cache_db)

        assert "Mood" in result.columns
        assert "Energy" in result.columns
        assert "Valence" in result.columns

    def test_batching_respects_batch_size(self, tmp_path):
        """With batch_size=2 and 4 tracks, exactly 2 HTTP calls are made."""
        mod = _import_module()
        cache_db = str(tmp_path / "batch_size.sqlite")

        df = pd.DataFrame(
            {
                "Artist": [f"Artist{i}" for i in range(4)],
                "Name": [f"Song{i}" for i in range(4)],
            }
        )

        batch_items = lambda offset: [
            {"idx": j + 1, "mood": "Chill", "energy": 5, "valence": 6, "tags": []}
            for j in range(2)
        ]

        resp1 = _make_ollama_response(batch_items(0))
        resp2 = _make_ollama_response(batch_items(2))

        with patch.object(mod.requests, "post", side_effect=[resp1, resp2]) as mock_post:
            mod.batch_enrich_ollama(df, cache_db=cache_db, batch_size=2)

        assert mock_post.call_count == 2

    def test_cache_db_is_created_in_tmp_path(self, tmp_path):
        """The SQLite cache file must be created at the path we specify."""
        mod = _import_module()
        cache_db = str(tmp_path / "subdir" / "enrich.sqlite")

        df = pd.DataFrame({"Artist": ["A"], "Name": ["S"]})
        items = [{"idx": 1, "mood": "Happy", "energy": 7, "valence": 8, "tags": []}]
        mock_resp = _make_ollama_response(items)

        with patch.object(mod.requests, "post", return_value=mock_resp):
            mod.batch_enrich_ollama(df, cache_db=cache_db)

        assert Path(cache_db).exists()

    def test_cache_schema_has_expected_table(self, tmp_path):
        """After a run the SQLite DB must contain the ollama_enrichment table."""
        mod = _import_module()
        cache_db = str(tmp_path / "schema_test.sqlite")

        df = pd.DataFrame({"Artist": ["A"], "Name": ["S"]})
        items = [{"idx": 1, "mood": "Sad", "energy": 3, "valence": 2, "tags": []}]
        mock_resp = _make_ollama_response(items)

        with patch.object(mod.requests, "post", return_value=mock_resp):
            mod.batch_enrich_ollama(df, cache_db=cache_db)

        conn = sqlite3.connect(cache_db)
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        conn.close()
        assert "ollama_enrichment" in tables

    def test_rate_limit_ms_triggers_sleep(self, tmp_path):
        """When rate_limit_ms > 0 and there is more than one batch, time.sleep is called.

        Execution order of time.time() calls across two batches:
          Batch 0: no rate-limit guard (last_batch_time == 0.0).
                   POST succeeds -> last_batch_time = time.time()   [call 1]
          Batch 1: rate-limit guard fires (last_batch_time > 0).
                   elapsed = time.time() - last_batch_time          [call 2]
                   elapsed < rate_limit/1000 -> time.sleep(wait)    [sleep 1]
                   POST succeeds -> last_batch_time = time.time()   [call 3]

        Supplying call-1=0.01 and call-2=0.01 makes elapsed=0.0, so
        wait = 0.5 - 0.0 = 0.5 s > 0, which triggers the sleep.
        """
        mod = _import_module()
        cache_db = str(tmp_path / "rate_limit.sqlite")

        df = pd.DataFrame(
            {
                "Artist": [f"Artist{i}" for i in range(4)],
                "Name": [f"Song{i}" for i in range(4)],
            }
        )

        resp = _make_ollama_response(
            [{"idx": j + 1, "mood": "Chill", "energy": 5, "valence": 5, "tags": []} for j in range(2)]
        )

        # Three time.time() calls total across two batches (see docstring above).
        time_sequence = [0.01, 0.01, 0.52]

        module_path = "playlistgen.enrichers.ollama_enricher"
        with patch.object(mod.requests, "post", return_value=resp):
            with patch(f"{module_path}.time.sleep") as mock_sleep:
                with patch(f"{module_path}.time.time", side_effect=time_sequence):
                    mod.batch_enrich_ollama(
                        df, cache_db=cache_db, batch_size=2, rate_limit_ms=500
                    )

        # sleep must have been called at least once for the rate-limit wait
        assert mock_sleep.call_count >= 1
        # Verify the sleep duration is approximately the expected wait time
        sleep_arg = mock_sleep.call_args_list[0][0][0]
        assert sleep_arg > 0
