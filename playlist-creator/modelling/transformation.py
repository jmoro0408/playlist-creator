import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from numpy import ravel
from imblearn.over_sampling import SMOTE

def prep_test_df():
    playlist_dir = r"/Users/jamesmoro/Documents/Python/playlist-recommender/playlist-creator/data/playlist_df.pkl"
    df = pd.read_pickle(playlist_dir)
    useless_cols = ["type", "id", "uri", "track_href", "analysis_url"]
    df = df.drop(useless_cols, axis = 1)
    X = df.drop('playlist_name', axis = 1)
    y = df['playlist_name']
    return train_test_split(X, y, test_size=0.2, random_state=0, stratify = y, shuffle = True)


class Transform():
    def __init__(self):
        pass

    def oversample(self, X, y):
        try:
            oversample = SMOTE()
            X, y = oversample.fit_resample(X, y)
            return X,y
        except ValueError:
            print("X,y must be encoded into numerical values before oversampling")
            return None

class Encode():
    def __init__(self):
        pass

    def standardscaler(self, X):
        cat_features = X.select_dtypes(include = ['object']).columns.to_list()
        num_features = [x for x in X.columns if x not in cat_features]
        scaler =StandardScaler()
        scaler.fit_transform(X[num_features])
        return scaler

    def standardscaler_ohe(self, X):
        cat_features = X.select_dtypes(include = ['object']).columns.to_list()

        X[cat_features] = X[cat_features].astype("category")
        numeric_transformer =StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, selector(dtype_exclude="category")),
            ("cat", categorical_transformer, selector(dtype_include="category")),
        ])
        preprocessor.fit_transform(X)
        return preprocessor

    def label_encode_y(self, y):
        y = y.values
        label_encoder = LabelEncoder()
        label_encoder.fit_transform(ravel(y))
        return label_encoder



if __name__ == "__main__":
    pass









