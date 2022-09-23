import os
import sys

import wandb

import itertools

from playlist_creator.modelling import model_pipeline
import numpy as np
import pandas as pd
from playlist_creator.modelling import utils
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, OneHotEncoder,
                                   RobustScaler, StandardScaler)


def test_transformation(X, y, config):
    pipe = model_pipeline.make_config_pipeline(X, config)
    scores = cross_val_score(pipe, X, y, cv=5, scoring="f1_macro")
    f1_score = np.mean(scores)
    print(f"F1 score: {f1_score:.3f}")
    return f1_score


X, y = utils.prep_playlist_df()

scalers = [None, StandardScaler(), MinMaxScaler(), RobustScaler(), MaxAbsScaler()]
samplers = [None, RandomOverSampler()]
featurisers = [OneHotEncoder(handle_unknown="ignore")]  # None doesn't work well
classifiers = [
    LogisticRegression(max_iter=1200),
    MLPClassifier(max_iter=1000),
    RandomForestClassifier(),
]

total_config = len(scalers) * len(samplers) * len(featurisers) * len(classifiers)

config_permuation_builer = {
    "scaler": scalers,
    "sampler": samplers,
    "featuriser": featurisers,
    "classifier": classifiers,
}
_keys, _values = zip(*config_permuation_builer.items())
config_permutations = [dict(zip(_keys, v)) for v in itertools.product(*_values)]

wandb.init(
    project="spotify-recommender", tags=["transform"], name="transformer comparisons"
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
    _values_list, columns=["scaler", "sampler", "featuriser", "classifier", "f1"]
)
wandb.log({"config": config_df})
wandb.finish()
