import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pgen.normalize import track_key  # noqa: E402


def test_canon_basic():
    assert track_key("The Beatles", "Let It Be (Remastered 2009)") == ("beatles", "let it be")
    assert track_key("A. Keys", "Fallin' feat. John Mayer") == ("a keys", "fallin")
    assert track_key("Mötley Crüe", "Kickstart My Heart") == ("motley crue", "kickstart my heart")
    assert track_key("Simon & Garfunkel", "The Boxer") == ("simon and garfunkel", "boxer")


def test_canon_empty():
    assert track_key(None, None) == ("", "")
    assert track_key("", "  ") == ("", "")
