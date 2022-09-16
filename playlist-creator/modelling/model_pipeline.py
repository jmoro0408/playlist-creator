import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prep_playlist_df():
    playlist_dir = r"/Users/jamesmoro/Documents/Python/playlist-recommender/playlist-creator/data/playlist_df.pkl"
    df = pd.read_pickle(playlist_dir)
    useless_cols = ["type", "id", "uri", "track_href", "analysis_url"]
    df = df.drop(useless_cols, axis=1)
    X = df.drop("playlist_name", axis=1)
    y = df["playlist_name"]
    return X, y


def make_model_pipeline(X):
    cat_features = X.select_dtypes(include=["object"]).columns.to_list()
    num_features = [x for x in X.columns if x not in cat_features]
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    oversampler = RandomOverSampler()
    classifier = RandomForestClassifier()
    featurisation = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
        ]
    )
    pipeline = imbPipeline(
        [
            ("features", featurisation),
            ("sampler", oversampler),
            ("classifier", classifier),
        ]
    )
    return pipeline


if __name__ == "__main__":
    pass
