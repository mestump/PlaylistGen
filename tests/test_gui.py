import playlistgen.gui as gui


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
