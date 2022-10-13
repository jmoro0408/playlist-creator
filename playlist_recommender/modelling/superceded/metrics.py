import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics


def accuracy(y_test, y_pred) -> float:
    """Returns the accuracy score as calculated by sklearn

    Args:
        y_test (array): true y labels
        y_pred (array): predicted y labels

    Returns:
        float: accuracy score between 0 and 1
    """
    return metrics.accuracy_score(y_test, y_pred)


def precision(y_test, y_pred, average="macro"):
    """
    average = 'micro' or 'macro'
    Macro averaged precision:
    calculate precision for all classes individually and then average them
    Micro averaged precision: calculate class wise true positive and false
    positive and then use that to calculate overall precision
    """
    return metrics.precision_score(y_test, y_pred, average=average)


def recall(y_test, y_pred, average="macro"):
    return metrics.recall_score(y_test, y_pred, average=average)


def f1(y_test, y_pred, average="macro"):
    return metrics.f1_score(y_test, y_pred, average=average)


def confusion_matrix(y_test, y_pred):
    plt.figure(figsize=(18, 12))
    sns.heatmap(
        metrics.confusion_matrix(y_test, y_pred),
        annot=True,
        xticklabels=np.unique(y_test),
        yticklabels=np.unique(y_test),
        cmap="YlGnBu",
    )
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()


def print_all_metrics(y_test, y_pred):
    print(f"accuracy: {accuracy(y_test, y_pred):.3f}")
    # print(f"precision: {precision(y_test, y_pred):.3f}")
    print(f"recall: {recall(y_test, y_pred):.3f}")
    print(f"f1: {f1(y_test, y_pred):.3f}")
    print(f"roc auc: {roc_auc_score_multiclass(y_test, y_pred):.3f}")
    confusion_matrix(y_test, y_pred)
