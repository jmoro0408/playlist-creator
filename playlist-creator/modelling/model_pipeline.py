import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDClassifier
class ClfSwitcher(BaseEstimator):

    def __init__(
        self,
        estimator = SGDClassifier(),
    ):
        """
        A Custom BaseEstimator that can switch between classifiers.
        :param estimator: sklearn object - The classifier
        """

        self.estimator = estimator

    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def score(self, X, y):
        return self.estimator.score(X, y)



def make_random_forest_pipeline(X):
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

def make_model_pipeline(X):
    cat_features = X.select_dtypes(include=["object"]).columns.to_list()
    num_features = [x for x in X.columns if x not in cat_features]
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    oversampler = RandomOverSampler()
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
            ('clf', ClfSwitcher()),
        ]
    )
    return pipeline

if __name__ == "__main__":
    pass