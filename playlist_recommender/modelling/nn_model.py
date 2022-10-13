import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from keras import layers
from keras.callbacks import EarlyStopping
from playlist_recommender.modelling import model_pipeline, utils
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras


def make_X_y() -> Pipeline:
    X, y = utils.prep_playlist_df()
    return model_pipeline.make_best_transformation_pipeline(X, y)


def label_encode_y(
    y_train: np.ndarray, y_test: np.ndarray, save_encoding_json: bool = True
):
    le = LabelEncoder()
    le.fit(y_train)
    if save_encoding_json:
        le_dict = dict(zip(le.transform(le.classes_), le.classes_))
        le_dict_json = {str(key): str(value) for key, value in le_dict.items()}
        with open("label_encoding.json", "w", encoding="UTF-8") as fp:
            json.dump(le_dict_json, fp)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    return y_train, y_test


def get_class_weights(y_train: np.ndarray) -> dict:
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    return dict(enumerate(class_weights))


def build_model(output_layer_size: int, fc_layer_size: int = 15) -> keras.Sequential:
    """generate a dense neural net with 5 layers with neurons
    of fc_layer_size

    Args:
        fc_layer_size (int, optional): number of neurons per layer. Defaults to 15.

    Returns:
        Keras.Sequential: Keras sequential model
    """
    input_shape = X_train.shape[1]
    return keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Dense(fc_layer_size, activation="relu"),
            layers.Dense(fc_layer_size, activation="relu"),
            layers.Dense(fc_layer_size, activation="relu"),
            layers.Dense(fc_layer_size, activation="relu"),
            layers.Dense(output_layer_size, activation="softmax"),
        ]
    )


def train(config: dict) -> np.ndarray:
    """Trains the neural net as built in build_model and parameters
    given in the config argument.

    Args:
        config (dict): configuration of model, should include:
            "batch_size": batch size
            "epochs": epochs to train for
            "fc_layer_size": fully connected layer size - no. neurons per layer
            "learning_rate": learning rate
    Returns:
        np.ndarray: predictions array
    """
    keras.backend.clear_session()
    model = build_model(
        output_layer_size=config["output_layer_size"],
        fc_layer_size=config["fc_layer_size"],
    )

    opt = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
    model.compile(opt, "sparse_categorical_crossentropy", metrics=["acc"])
    es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=200)

    model.fit(
        X_train,
        y_train,
        epochs=config["epochs"],
        validation_data=(X_test, y_test),
        class_weight=class_weight_dict,
        callbacks=[es],
    )

    model.save("trained_nn_model")
    y_probas = model.predict(X_test)
    y_pred = tf.argmax(y_probas, axis=-1)
    return y_pred


def plot_confusion_matrix(
    y_test: np.ndarray, y_pred: np.ndarray, encoding_dict: dict, save=True
) -> None:
    """creates confusion matrix of predictions

    Args:
        y_test (np.ndarray): true y labels
        y_pred (np.ndarray): predicted y labels
        encoding_dict (dict): dictionary of y-label encoding
        save (bool, optional): save matrix .png file. Defaults to True.
    """
    cm = tf.math.confusion_matrix(y_test, y_pred)
    cm = cm / cm.numpy().sum(axis=1)[:, tf.newaxis]
    plt.figure(figsize=(26, 18))
    sns.heatmap(
        cm,
        annot=True,
        xticklabels=encoding_dict.values(),
        yticklabels=encoding_dict.values(),
        cmap="YlGnBu",
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if save:
        plt.savefig("confusion_matrix.png", facecolor="white", format="png")
    f1_score = metrics.f1_score(y_test, y_pred, average="macro", zero_division=0)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f"F1 Score: {f1_score}")
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = make_X_y()
    y_train, y_test = label_encode_y(y_train, y_test)
    class_weight_dict = get_class_weights(y_train)
    parameter_config = {
        "batch_size": 32,
        "epochs": 550,
        "fc_layer_size": 256,
        "learning_rate": 1e-05,
        "output_layer_size": len(class_weight_dict),
    }

    y_pred = train(config=parameter_config)
    plot_confusion_matrix(y_test, y_pred, encoding_dict=class_weight_dict)
