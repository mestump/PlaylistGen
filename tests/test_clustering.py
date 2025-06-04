import pandas as pd
from playlistgen.clustering import name_cluster, humanize_label


def test_humanize_label_basic():
    assert humanize_label("Sad", "Country") == "Melancholic Country"
    assert humanize_label("Happy", None) == "Joyful"
    assert humanize_label(None, "Rock") == "Rock"


def test_name_cluster_human_order():
    df = pd.DataFrame({"Mood": ["Sad", "Sad"], "Genre": ["Country", "Country"]})
    assert name_cluster(df) == "Melancholic Country"


