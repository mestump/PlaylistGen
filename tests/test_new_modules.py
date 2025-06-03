import pandas as pd
from playlistgen import analyze_playlists, score_playlists, vectorize_playlists


def test_analyze_and_score():
    p1 = pd.DataFrame({"Genre": ["Rock", "Rock"], "Mood": ["Happy", "Happy"]})
    p2 = pd.DataFrame({"Genre": ["Pop"], "Mood": ["Chill"]})
    patterns = analyze_playlists([p1, p2], n_clusters=2)
    assert "labels" in patterns
    bench_vecs, vec = vectorize_playlists([p1])
    scores = score_playlists([p1, p2], [bench_vecs[0]], vectorizer=vec)
    assert len(scores) == 2
