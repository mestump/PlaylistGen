import unittest
import pandas as pd
from playlistgen.clustering import name_cluster, MOOD_ADJECTIVES

class TestNameCluster(unittest.TestCase):
    def test_name_with_adjective_and_genre(self):
        df = pd.DataFrame({'Mood': ['Sad'], 'Genre': ['Country']})
        self.assertEqual(name_cluster(df), "Melancholic Country Mix")

    def test_name_with_mood_no_adjective_and_genre(self):
        df = pd.DataFrame({'Mood': ['Quirky'], 'Genre': ['Pop']})
        # Assuming "Quirky" is not in MOOD_ADJECTIVES
        self.assertEqual(name_cluster(df), "Quirky Pop Mix")

    def test_name_with_genre_only(self):
        df = pd.DataFrame({'Genre': ['Rock']})
        self.assertEqual(name_cluster(df), "Rock Mix")

    def test_name_with_mood_adjective_only(self):
        df = pd.DataFrame({'Mood': ['Happy']})
        self.assertEqual(name_cluster(df), "Joyful Mix")

    def test_name_with_mood_no_adjective_only(self):
        df = pd.DataFrame({'Mood': ['Thoughtful']})
        # Assuming "Thoughtful" is not in MOOD_ADJECTIVES
        self.assertEqual(name_cluster(df), "Thoughtful Mix")

    def test_name_no_mood_no_genre_with_index(self):
        df = pd.DataFrame()
        self.assertEqual(name_cluster(df, i=0), "Cluster 1")
        
        df_with_cols_no_data = pd.DataFrame({'Mood': [], 'Genre': []})
        self.assertEqual(name_cluster(df_with_cols_no_data, i=0), "Cluster 1")

        df_with_none_data = pd.DataFrame({'Mood': [None], 'Genre': [None]})
        self.assertEqual(name_cluster(df_with_none_data, i=0), "Cluster 1")


    def test_name_no_mood_no_genre_no_index(self):
        df = pd.DataFrame()
        self.assertEqual(name_cluster(df), "Cluster")

    def test_name_case_insensitivity(self):
        df = pd.DataFrame({'Mood': ['sAd'], 'Genre': ['counTRY']})
        self.assertEqual(name_cluster(df), "Melancholic Country Mix")
        
        df_lower_mood = pd.DataFrame({'Mood': ['happy'], 'Genre': ['ElecTRONic']})
        self.assertEqual(name_cluster(df_lower_mood), "Joyful Electronic Mix")

    def test_name_multiple_moods_genres(self):
        df = pd.DataFrame({
            'Mood': ['Sad', 'Sad', 'Happy', 'Energetic', 'Sad'], 
            'Genre': ['Country', 'Rock', 'Country', 'Pop', 'Country']
        })
        # Mode for Mood is 'Sad', mode for Genre is 'Country'
        self.assertEqual(name_cluster(df), "Melancholic Country Mix")

    def test_empty_dataframe_with_index(self):
        df = pd.DataFrame()
        self.assertEqual(name_cluster(df, i=0), "Cluster 1")

    def test_mood_column_missing(self):
        df = pd.DataFrame({'Genre': ['Electronic']})
        self.assertEqual(name_cluster(df, i=0), "Electronic Mix")

    def test_genre_column_missing(self):
        df = pd.DataFrame({'Mood': ['Chill']})
        self.assertEqual(name_cluster(df, i=0), "Relaxed Mix")

    def test_mood_column_all_nan(self):
        df = pd.DataFrame({'Mood': [None, None], 'Genre': ['Rock', 'Rock']}) # Corrected length
        self.assertEqual(name_cluster(df), "Rock Mix")

    def test_genre_column_all_nan(self):
        df = pd.DataFrame({'Mood': ['Happy', 'Happy'], 'Genre': [None, None]}) # Corrected length
        self.assertEqual(name_cluster(df), "Joyful Mix")

    def test_mood_and_genre_columns_all_nan(self):
        df = pd.DataFrame({'Mood': [None, None], 'Genre': [None, None]})
        self.assertEqual(name_cluster(df, i=0), "Cluster 1")
        self.assertEqual(name_cluster(df), "Cluster")
        
    def test_mood_adjective_lookup_various_cases(self):
        # Test a few specific MOOD_ADJECTIVES lookups
        df_energetic = pd.DataFrame({'Mood': ['Energetic'], 'Genre': ['Pop']})
        self.assertEqual(name_cluster(df_energetic), "Upbeat Pop Mix")

        df_spacey = pd.DataFrame({'Mood': ['Spacey']})
        self.assertEqual(name_cluster(df_spacey), "Ethereal Mix")
        
        df_romantic_genre = pd.DataFrame({'Mood': ['romantic'], 'Genre': ['Jazz']})
        self.assertEqual(name_cluster(df_romantic_genre), "Romantic Jazz Mix")

if __name__ == '__main__':
    unittest.main()
