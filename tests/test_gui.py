import playlistgen.gui as gui
import sys


def test_run_gui_generate_mix(monkeypatch):
    answers = iter(["Generate mix", "", ""])

    def fake_select(*args, **kwargs):
        class Q:
            def ask(self):
                return next(answers)
        return Q()

    def fake_text(*args, **kwargs):
        class Q:
            def ask(self):
                return next(answers)
        return Q()

    called = {}
    def fake_run_pipeline(cfg, genre=None, mood=None, library_dir=None):
        called["genre"] = genre
        called["mood"] = mood

    monkeypatch.setattr(gui.questionary, "select", fake_select)
    monkeypatch.setattr(gui.questionary, "text", fake_text)
    monkeypatch.setattr(gui, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(gui, "load_config", lambda: {})

    action = gui.run_gui()
    assert action == "Generate mix"
    assert called == {"genre": None, "mood": None}


def test_run_gui_seed_song(monkeypatch):
    answers = iter(["Generate from seed song", "Artist - Title", "5"])

    def fake_select(*args, **kwargs):
        class Q:
            def ask(self):
                return next(answers)
        return Q()

    def fake_text(*args, **kwargs):
        class Q:
            def ask(self):
                return next(answers)
        return Q()

    called = {}
    def fake_build(song, cfg=None, library_dir=None, limit=20):
        called["song"] = song
        called["limit"] = limit

    monkeypatch.setattr(gui.questionary, "select", fake_select)
    monkeypatch.setattr(gui.questionary, "text", fake_text)
    monkeypatch.setattr(gui, "build_seed_playlist", fake_build)
    monkeypatch.setattr(gui, "load_config", lambda: {})

    action = gui.run_gui()
    assert action == "Generate from seed song"
    assert called == {"song": "Artist - Title", "limit": 5}


def test_spotify_login_uses_redirect_from_config(monkeypatch):
    captured = {}

    class FakeAuth:
        def __init__(self, client_id=None, client_secret=None, scope=None, redirect_uri=None):
            captured["redirect_uri"] = redirect_uri

        def get_access_token(self, as_dict=False):
            return "tok"

    import types

    fake_spotipy = types.ModuleType("spotipy")
    fake_oauth2 = types.ModuleType("oauth2")
    fake_oauth2.SpotifyOAuth = FakeAuth
    fake_spotipy.oauth2 = fake_oauth2

    monkeypatch.setitem(sys.modules, "spotipy", fake_spotipy)
    monkeypatch.setitem(sys.modules, "spotipy.oauth2", fake_oauth2)
    monkeypatch.setattr(gui, "save_config", lambda cfg: None)

    cfg = {
        "SPOTIFY_CLIENT_ID": "id",
        "SPOTIFY_CLIENT_SECRET": "secret",
        "SPOTIFY_REDIRECT_URI": "http://127.0.0.1:8888/callback",
    }

    gui.spotify_login(cfg)
    assert captured["redirect_uri"] == "http://127.0.0.1:8888/callback"
