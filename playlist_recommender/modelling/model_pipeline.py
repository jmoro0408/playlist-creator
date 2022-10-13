import joblib
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder


class ClfSwitcher(BaseEstimator):
    """Custom class to allow switching between classifiers

    Args:
        BaseEstimator: sklearn estimator
    """

    def __init__(
        self,
        estimator=SGDClassifier(),
    ):
        """
        A Custom BaseEstimator that can switch between classifiers.
        :param estimator: sklearn object - The classifier
        """

        self.estimator = estimator

    def fit(self, X, y=None):
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def score(self, X, y):
        return self.estimator.score(X, y)


def make_config_pipeline(X: pd.DataFrame, config: dict):
    """Builds and returns an sklearn pipeline with features as defined
    in the config dictionary argument

    Args:
        X (pd.DataFrame): training data array
        config (dict): configuration dict with scaler, sampler,
        featuriser, and classifier defined

    Returns:
        Pipeline: sklearn pipeline of the required config components
    """
    scaler = config.get("scaler")
    sampler = config.get("sampler")
    featuriser = config.get("featuriser")
    clf = config.get("classifier")

    cat_features = X.select_dtypes(include=["object"]).columns.to_list()
    num_features = [x for x in X.columns if x not in cat_features]

    if (scaler is not None) and (featuriser is not None):
        featurisation = ColumnTransformer(
            transformers=[
                ("num", scaler, num_features),
                ("cat", featuriser, cat_features),
            ]
        )

    elif scaler is None:
        featurisation = ColumnTransformer(
            transformers=[("cat", featuriser, cat_features)]
        )
    elif featuriser is None:
        featurisation = ColumnTransformer(transformers=[("num", scaler, num_features)])

    if sampler is not None:
        pipeline = imbPipeline(
            [
                ("features", featurisation),
                ("sampler", sampler),
                ("clf", clf),
            ]
        )
        return pipeline

    if sampler is None:
        pipeline = Pipeline(
            [
                ("features", featurisation),
                ("clf", clf),
            ]
        )

    return pipeline


def make_best_transformation_pipeline(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.35, oversample: bool = False
):
    """Creates a pipeline using the highest performing features as discovered through
    the transformation comparison modelling.
    This includes:
    1. MaxAbsScaler of numeric values
    2. OneHotEncoding of categorical values

    By default, no oversampling is applied as better results were seen when
    class weights are used to handle dataset imbalances.

    Args:
        X (pd.DataFrame): training features
        y (pd.Series): training labels
        test_size (float, optional): Ratio for test/train split. Defaults to 0.35.
        oversample (bool, optional): Whether to use oversampling or not. Defaults to False.

    Returns:
        Pipeline: sklearn pipeline
    """
    cat_features = X.select_dtypes(include=["object"]).columns.to_list()
    num_features = [x for x in X.columns if x not in cat_features]
    if oversample:
        oversample = RandomOverSampler()
        X, y = oversample.fit_resample(X, y)
    numeric_transformer = MaxAbsScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    featurisation = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
        ]
    )
    pipeline = Pipeline(
        [
            ("features", featurisation),
        ]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0, stratify=y, shuffle=True
    )
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, "pipeline.pkl")
    print("pipeline dumped")
    X_train = pipeline.transform(X_train)
    X_test = pipeline.transform(X_test)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    pass
