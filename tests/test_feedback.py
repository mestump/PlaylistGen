"""Tests for playlistgen.feedback — user feedback persistence."""

import json

import pytest

from playlistgen.feedback import load_feedback, save_feedback, update_feedback


class TestLoadFeedback:
    def test_load_existing_file(self, tmp_path):
        f = tmp_path / "feedback.json"
        f.write_text(json.dumps({"Mix A": ["like"]}))
        result = load_feedback(str(f))
        assert result == {"Mix A": ["like"]}

    def test_load_missing_file(self, tmp_path):
        result = load_feedback(str(tmp_path / "nonexistent.json"))
        assert result == {}


class TestSaveFeedback:
    def test_creates_file(self, tmp_path):
        f = tmp_path / "feedback.json"
        save_feedback(str(f), {"Mix": ["skip"]})
        assert f.exists()
        data = json.loads(f.read_text())
        assert data == {"Mix": ["skip"]}

    def test_creates_parent_dirs(self, tmp_path):
        f = tmp_path / "sub" / "dir" / "feedback.json"
        save_feedback(str(f), {})
        assert f.exists()


class TestUpdateFeedback:
    def test_appends_action(self, tmp_path):
        f = tmp_path / "feedback.json"
        f.write_text(json.dumps({"Mix A": ["like"]}))
        update_feedback(str(f), "Mix A", "skip")
        data = json.loads(f.read_text())
        assert data["Mix A"] == ["like", "skip"]

    def test_creates_new_playlist_entry(self, tmp_path):
        f = tmp_path / "feedback.json"
        f.write_text(json.dumps({}))
        update_feedback(str(f), "New Mix", "like")
        data = json.loads(f.read_text())
        assert data["New Mix"] == ["like"]

    def test_creates_file_if_missing(self, tmp_path):
        f = tmp_path / "feedback.json"
        update_feedback(str(f), "Mix", "like")
        assert f.exists()
        data = json.loads(f.read_text())
        assert data == {"Mix": ["like"]}
