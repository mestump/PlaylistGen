"""Tests for playlistgen.prompt_io — prompt generation and response parsing."""

import json

import pandas as pd
import pytest

from playlistgen.prompt_io import (
    _format_enrich_line,
    _format_curate_line,
    _extract_json_from_text,
    _extract_response_from_file,
    _detect_mode,
)


# ---------------------------------------------------------------------------
# _format_enrich_line
# ---------------------------------------------------------------------------


class TestFormatEnrichLine:
    def test_basic_format(self):
        row = pd.Series({"Artist": "Radiohead", "Name": "Creep", "Genre": "Rock", "BPM": 92, "Year": 1993})
        result = _format_enrich_line(1, row)
        assert result == "1. Radiohead — Creep | Genre:Rock | BPM:92 | Year:1993"

    def test_missing_genre(self):
        row = pd.Series({"Artist": "X", "Name": "Y", "Genre": None, "BPM": 120, "Year": 2020})
        result = _format_enrich_line(1, row)
        assert "Genre:" not in result
        assert "BPM:120" in result

    def test_missing_bpm(self):
        row = pd.Series({"Artist": "X", "Name": "Y", "Genre": "Pop", "BPM": None, "Year": 2020})
        result = _format_enrich_line(1, row)
        assert "BPM:" not in result

    def test_zero_bpm_excluded(self):
        row = pd.Series({"Artist": "X", "Name": "Y", "Genre": "Pop", "BPM": 0, "Year": 2020})
        result = _format_enrich_line(1, row)
        assert "BPM:" not in result

    def test_missing_year(self):
        row = pd.Series({"Artist": "X", "Name": "Y", "Genre": "Pop", "BPM": 100})
        result = _format_enrich_line(1, row)
        assert "Year:" not in result

    def test_unknown_artist_fallback(self):
        row = pd.Series({"Artist": None, "Name": "Track"})
        result = _format_enrich_line(1, row)
        assert "Unknown Artist" in result


# ---------------------------------------------------------------------------
# _format_curate_line
# ---------------------------------------------------------------------------


class TestFormatCurateLine:
    def test_includes_mood_and_energy(self):
        row = pd.Series({
            "Artist": "Daft Punk", "Name": "Get Lucky",
            "Mood": "Groovy", "Genre": "Electronic", "Energy": 0.7, "BPM": 116, "Year": 2013,
        })
        result = _format_curate_line(1, row)
        assert "Mood:Groovy" in result
        assert "Energy:1" in result  # round(0.7) = 1
        assert "Genre:Electronic" in result

    def test_unknown_mood_excluded(self):
        row = pd.Series({"Artist": "X", "Name": "Y", "Mood": "Unknown", "Genre": "Pop"})
        result = _format_curate_line(1, row)
        assert "Mood:" not in result

    def test_era_label_instead_of_year(self):
        row = pd.Series({"Artist": "X", "Name": "Y", "Year": 1985})
        result = _format_curate_line(1, row)
        assert "Era:1985" in result
        assert "Year:" not in result


# ---------------------------------------------------------------------------
# _extract_json_from_text
# ---------------------------------------------------------------------------


class TestExtractJsonFromText:
    def test_raw_json_array(self):
        raw = '[{"mood": "Happy"}]'
        assert _extract_json_from_text(raw) == raw

    def test_raw_json_object(self):
        raw = '{"name": "test"}'
        assert _extract_json_from_text(raw) == raw

    def test_markdown_code_fence(self):
        raw = '```json\n[{"mood": "Sad"}]\n```'
        result = _extract_json_from_text(raw)
        assert result == '[{"mood": "Sad"}]'

    def test_plain_code_fence(self):
        raw = '```\n{"key": "val"}\n```'
        result = _extract_json_from_text(raw)
        assert result == '{"key": "val"}'

    def test_json_after_prose(self):
        raw = 'Here is the result:\n[{"mood": "Chill"}]'
        result = _extract_json_from_text(raw)
        parsed = json.loads(result)
        assert parsed[0]["mood"] == "Chill"

    def test_nested_json(self):
        raw = '{"playlists": [{"name": "Mix", "tracks": [1, 2]}]}'
        result = _extract_json_from_text(raw)
        parsed = json.loads(result)
        assert "playlists" in parsed

    def test_no_json_returns_text(self):
        raw = "Just plain text with no JSON"
        result = _extract_json_from_text(raw)
        assert result == raw


# ---------------------------------------------------------------------------
# _extract_response_from_file
# ---------------------------------------------------------------------------


class TestExtractResponseFromFile:
    def test_extracts_between_markers(self):
        text = (
            "Some header\n"
            "── RESPONSE START ────────────\n"
            '[{"mood": "Happy"}]\n'
            "── RESPONSE END ──────────────\n"
        )
        result = _extract_response_from_file(text)
        assert result == '[{"mood": "Happy"}]'

    def test_raw_json_file(self):
        text = '[{"mood": "Sad"}]'
        result = _extract_response_from_file(text)
        assert result == text

    def test_no_markers_returns_text(self):
        text = "Just some text"
        result = _extract_response_from_file(text)
        assert result == text


# ---------------------------------------------------------------------------
# _detect_mode
# ---------------------------------------------------------------------------


class TestDetectMode:
    def test_detects_enrich_from_header(self):
        text = "# PlaylistGen Prompt\nMode: enrich\n"
        assert _detect_mode(text) == "enrich"

    def test_detects_curate_from_header(self):
        text = "# PlaylistGen Prompt\nMode: curate\n"
        assert _detect_mode(text) == "curate"

    def test_guesses_enrich_from_json_array(self):
        text = '[{"mood": "Happy", "energy": 7}]'
        assert _detect_mode(text) == "enrich"

    def test_guesses_curate_from_json_object(self):
        text = '{"playlists": [{"name": "Mix"}]}'
        assert _detect_mode(text) == "curate"
