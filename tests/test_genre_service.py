# genre_service.py was deleted and its functionality merged into lastfm_client.py.
# See test_lastfm_client.py for the replacement tests.

def test_genre_service_removed():
    """Confirm genre_service.py no longer exists as a separate module."""
    import importlib
    import importlib.util
    spec = importlib.util.find_spec("playlistgen.genre_service")
    assert spec is None, (
        "playlistgen.genre_service should be gone — "
        "all genre functionality is in lastfm_client.py"
    )
