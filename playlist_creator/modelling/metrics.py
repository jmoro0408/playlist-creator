import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics


def accuracy(y_test, y_pred):
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


def roc_auc_score_multiclass(actual_class, pred_class, average="macro"):

    # creating a set of all the unique classes using the actual class list
    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:

        # creating a list of all the classes except the current class
        other_class = [x for x in unique_class if x != per_class]

        # marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        # using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = metrics.roc_auc_score(
            new_actual_class, new_pred_class, average=average
        )
        roc_auc_dict[per_class] = roc_auc

    return roc_auc_dict


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
