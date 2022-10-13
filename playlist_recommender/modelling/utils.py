from pathlib import Path

import pandas as pd


def prep_playlist_df() -> tuple([pd.DataFrame, pd.Series]):
    """reads in playlist df, drops unused columns and returns X,y
    features for further processing

    Returns:
        tuple([pd.DataFrame, pd.Series]): X,y
    """
    playlist_dir = Path(Path.cwd().parent, "data", "playlist_df.pkl")
    df = pd.read_pickle(playlist_dir)
    useless_cols = ["type", "id", "uri", "track_href", "analysis_url", "track_names"]
    df = df.drop(useless_cols, axis=1)
    X = df.drop("playlist_name", axis=1)
    y = df["playlist_name"]
    return X, y


if __name__ == "__main__":
    pass
