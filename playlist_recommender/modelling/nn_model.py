import numpy as np
import wandb

from playlist_recommender.modelling import model_pipeline
from playlist_recommender.modelling import utils
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from wandb.keras import WandbCallback
import tensorflow as tf
from tensorflow import keras
from keras import layers
import seaborn as sns
import matplotlib.pyplot as plt

X, y = utils.prep_playlist_df()
X_train, X_test, y_train, y_test = model_pipeline.make_best_transformation_pipeline(
    X, y
)
le = LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train)
y_test = le.transform(y_test)
le_dict = dict(zip(le.transform(le.classes_),le.classes_))

parameter_config = {'batch_size': 32,
                'epochs': 3,
                'fc_layer_size': 256,
                'learning_rate': 1e-05}

def build_model(fc_layer_size=15):
    input_shape = X_train.shape[1]
    num_classes = len(le.classes_)
    return keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Dense(fc_layer_size, activation="relu"),
            layers.Dense(fc_layer_size, activation="relu"),
            layers.Dense(fc_layer_size, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

def train(config = parameter_config):
    keras.backend.clear_session()
    model = build_model(fc_layer_size=config['fc_layer_size'])

    # Compile the model
    opt = tf.keras.optimizers.Adam(
        learning_rate=config['learning_rate']
    )  # optimizer with different learning rate specified by config
    model.compile(opt, "sparse_categorical_crossentropy", metrics=["acc"])

    # Train the model
    model.fit(
        X_train,
        y_train,
        epochs=config['epochs'],
        validation_data=(X_test, y_test))

    model.save('playlist_recommender/modelling/trained_nn_model')
    y_probas=model.predict(X_test)
    y_pred = tf.argmax(y_probas, axis=-1)
    f1_score = metrics.f1_score(y_test, y_pred, average="macro", zero_division=0)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    recall = metrics.recall_score(y_test, y_pred, average="macro")
    y_pred_names = le.inverse_transform(y_pred)
    y_test_names= le.inverse_transform(y_test)

    cm = tf.math.confusion_matrix(y_test, y_pred)
    cm = cm/cm.numpy().sum(axis=1)[:, tf.newaxis]
    plt.figure(figsize=(18, 12))
    sns.heatmap(
        cm, annot=True,
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        cmap="YlGnBu",)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("playlist_recommender/modelling/confusion_matrix.png", facecolor='white', format = 'png')


if __name__ == "__main__":
    train()