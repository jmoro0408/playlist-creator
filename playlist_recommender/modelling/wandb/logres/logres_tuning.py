import os
import sys

import wandb

from playlist_recommender.modelling import model_pipeline
import numpy as np
import pandas as pd
from playlist_recommender.modelling import utils
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MaxAbsScaler
