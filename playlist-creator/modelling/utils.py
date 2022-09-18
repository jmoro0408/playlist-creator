import pandas as pd


def prep_playlist_df():
    playlist_dir = r"/Users/jamesmoro/Documents/Python/playlist-recommender/playlist-creator/data/playlist_df.pkl"
    df = pd.read_pickle(playlist_dir)
    useless_cols = ["type", "id", "uri", "track_href", "analysis_url", "track_names"]
    df = df.drop(useless_cols, axis=1)
    X = df.drop("playlist_name", axis=1)
    y = df["playlist_name"]
    return X, y


def invert_label_encoding(le_fitted):
    return dict(zip(le_fitted.transform(le_fitted.classes_), le_fitted.classes_))


if __name__ == "__main__":
    pass
