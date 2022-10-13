"""
Compares logisitic regression and random forest classifiers
for various different transformations including:
1. standard scalar
2. maxabs scalar
3. one hot encoding

with tracking with weights and biases
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, OneHotEncoder,
                                   RobustScaler, StandardScaler)
from sklearn.utils import compute_class_weight

import wandb
from playlist_recommender.modelling import model_pipeline, utils


def test_transformation(X, y, config):
    pipe = model_pipeline.make_config_pipeline(X, config)
    scores = cross_val_score(pipe, X, y, cv=5, scoring="f1_macro")
    f1_score = np.mean(scores)
    print(f"F1 score: {f1_score:.3f}")
    return f1_score


X, y = utils.prep_playlist_df()
class_weights = compute_class_weight(class_weight = 'balanced',
                                                 classes = np.unique(y),
                                                 y = y)
class_weight_dict_labels = dict(zip(np.unique(y), class_weights))

scalers = [None, StandardScaler(), MinMaxScaler(), RobustScaler(), MaxAbsScaler()]
featurisers = [OneHotEncoder(handle_unknown="ignore")]
classifiers = [
    LogisticRegression(max_iter=1200,class_weight = class_weight_dict_labels),
    RandomForestClassifier(class_weight = class_weight_dict_labels),
]

total_config = len(scalers)  * len(featurisers) * len(classifiers)

config_permuation_builer = {
    "scaler": scalers,
    "featuriser": featurisers,
    "classifier": classifiers,
}
_keys, _values = zip(*config_permuation_builer.items())
config_permutations = [dict(zip(_keys, v)) for v in itertools.product(*_values)]

wandb.init(
    project="spotify-recommender", tags=["transform"], name="transformer comparisons - class weights"
)

_values_list = []
for idx, config in enumerate(config_permutations):
    if config["featuriser"] is None:
        X = X.drop("artist_names", axis=1)
    f1_score = test_transformation(X, y, config)
    _values = [str(x) for x in config.values()]
    _values.append(f1_score)
    _values_list.append(_values)
    wandb.log({"f1_score": f1_score})
    print(f"{idx+1} of {total_config} configurations.")

config_df = pd.DataFrame(
    _values_list, columns=["scaler", "featuriser", "classifier", "f1"]
)
wandb.log({"config": config_df})
wandb.finish()
