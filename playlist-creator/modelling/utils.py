import pandas as pd

def prep_playlist_df():
    playlist_dir = r"/Users/jamesmoro/Documents/Python/playlist-recommender/playlist-creator/data/playlist_df.pkl"
    df = pd.read_pickle(playlist_dir)
    useless_cols = ["type", "id", "uri", "track_href", "analysis_url"]
    df = df.drop(useless_cols, axis=1)
    X = df.drop("playlist_name", axis=1)
    y = df["playlist_name"]
    return X, y

if __name__ == "__main__":
    pass