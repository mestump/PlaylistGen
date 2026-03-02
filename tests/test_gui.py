import playlistgen.gui as gui
import sys


def test_run_gui_generate_mix(monkeypatch):
    select_answers = iter(["generate_mix", "exit"])

    def fake_select(*args, **kwargs):
        class Q:
            def ask(self):
                return next(select_answers)
        return Q()

    # confirm: skip wizard (False), then "return to menu?" (False to exit)
    confirm_answers = iter([False, False])

    def fake_confirm(*args, **kwargs):
        class Q:
            def ask(self):
                return next(confirm_answers)
        return Q()

    called = {}
    def fake_run_pipeline(cfg, genre=None, mood=None, library_dir=None):
        called["genre"] = genre
        called["mood"] = mood

    monkeypatch.setattr(gui.questionary, "select", fake_select)
    monkeypatch.setattr(gui.questionary, "confirm", fake_confirm)
    monkeypatch.setattr(gui, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(gui, "load_config", lambda: {"ITUNES_JSON": "itunes_slimmed.json"})
    monkeypatch.setattr(gui, "_welcome_first_run", lambda cfg: False)

    action = gui.run_gui()
    assert action == "generate_mix"
    assert called == {"genre": None, "mood": None}


def test_run_gui_seed_song(monkeypatch):
    select_answers = iter(["seed", "exit"])
    text_answers = iter(["Artist - Title", "5"])

    def fake_select(*args, **kwargs):
        class Q:
            def ask(self):
                return next(select_answers)
        return Q()

    def fake_text(*args, **kwargs):
        class Q:
            def ask(self):
                return next(text_answers)
        return Q()

    def fake_confirm(*args, **kwargs):
        class Q:
            def ask(self):
                return False
        return Q()

    called = {}
    def fake_build(song, cfg=None, library_dir=None, limit=20):
        called["song"] = song
        called["limit"] = limit

    monkeypatch.setattr(gui.questionary, "select", fake_select)
    monkeypatch.setattr(gui.questionary, "text", fake_text)
    monkeypatch.setattr(gui.questionary, "confirm", fake_confirm)
    monkeypatch.setattr(gui, "build_seed_playlist", fake_build)
    monkeypatch.setattr(gui, "load_config", lambda: {"ITUNES_JSON": "itunes_slimmed.json"})
    monkeypatch.setattr(gui, "_welcome_first_run", lambda cfg: False)

    action = gui.run_gui()
    assert action == "seed"
    assert called == {"song": "Artist - Title", "limit": 5}
