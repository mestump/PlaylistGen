"""Tests for mood_map.py — keyword coverage, canonical_mood, canonical_genre."""

import pytest
from playlistgen.mood_map import (
    MOODS,
    GENRE_MOOD_FALLBACK,
    GENRE_NORMALIZE,
    PRIORITY,
    canonical_mood,
    canonical_genre,
    build_tag_counts,
)


# ---------------------------------------------------------------------------
# Sanity checks on the data
# ---------------------------------------------------------------------------

def test_all_priority_moods_in_moods():
    for m in PRIORITY:
        assert m in MOODS, f"Priority mood '{m}' missing from MOODS dict"


def test_moods_have_enough_keywords():
    for mood, kws in MOODS.items():
        assert len(kws) >= 20, f"Mood '{mood}' only has {len(kws)} keywords (need >= 20)"


# ---------------------------------------------------------------------------
# canonical_mood — tag-based matching
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tags,expected", [
    (["happy", "feel good", "upbeat"], "Happy"),
    (["sad", "melancholy"], "Sad"),
    (["chill", "relax"], "Chill"),
    (["energetic", "dance", "party"], "Energetic"),
    (["romantic", "love song"], "Romantic"),
    (["epic", "orchestral", "cinematic"], "Epic"),
    (["dreamy", "ethereal", "ambient"], "Dreamy"),
    (["groovy", "funky", "soul"], "Groovy"),
    (["nostalgia", "retro", "80s"], "Nostalgic"),
    (["aggressive", "rage", "brutal"], "Angry"),
])
def test_canonical_mood_tag_match(tags, expected):
    result = canonical_mood(tags)
    assert result == expected, f"Tags {tags!r} → expected {expected!r}, got {result!r}"


def test_canonical_mood_empty_tags_no_genre():
    # No tags, no genre → None
    assert canonical_mood([]) is None
    assert canonical_mood(None) is None


def test_canonical_mood_genre_fallback():
    # No tags → should fall back to genre
    result = canonical_mood([], genre="Metal")
    assert result == "Angry"

    result = canonical_mood([], genre="Jazz")
    assert result == "Chill"

    result = canonical_mood([], genre="Reggae")
    assert result in ("Chill", "Groovy")  # both valid


def test_canonical_mood_idf_weighting():
    # Build a tag_counts that makes "happy" very common (low IDF)
    tag_counts = {"happy": 10000, "melancholy": 1}
    # The rare "melancholy" tag should outweigh the common "happy"
    result = canonical_mood(["happy", "melancholy"], tag_counts=tag_counts)
    assert result == "Sad"


def test_canonical_mood_legacy_dict_tags_not_accepted():
    # canonical_mood expects List[str], not dicts — should not raise
    result = canonical_mood(None, genre="Pop")
    assert result == "Happy"


# ---------------------------------------------------------------------------
# canonical_genre
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tag,expected", [
    ("rock", "Rock"),
    ("hip-hop", "Hip-Hop"),
    ("hip hop", "Hip-Hop"),
    ("r&b", "R&B"),
    ("jazz", "Jazz"),
    ("classical", "Classical"),
    ("edm", "Electronic"),
    ("house", "Electronic"),
    ("lo-fi", "Lo-Fi"),
    ("lofi", "Lo-Fi"),
    ("folk", "Folk"),
    ("country", "Country"),
    ("metal", "Metal"),
    ("indie rock", "Indie"),
    ("alt rock", "Alternative"),
])
def test_canonical_genre_known_tags(tag, expected):
    assert canonical_genre(tag) == expected, f"Tag {tag!r} → expected {expected!r}"


def test_canonical_genre_unknown():
    assert canonical_genre("seen live") is None
    assert canonical_genre("") is None
    assert canonical_genre(None) is None


# ---------------------------------------------------------------------------
# build_tag_counts
# ---------------------------------------------------------------------------

def test_build_tag_counts_list_format():
    db = {
        "artist a - track 1": ["rock", "indie", "happy"],
        "artist b - track 2": ["rock", "sad"],
    }
    counts = build_tag_counts(db)
    assert counts["rock"] == 2
    assert counts["indie"] == 1
    assert counts["sad"] == 1


def test_build_tag_counts_legacy_dict_format():
    db = {
        "artist a - track 1": {"tags": ["rock", "indie"], "mood": "Happy"},
    }
    counts = build_tag_counts(db)
    assert counts["rock"] == 1
    assert counts["indie"] == 1
