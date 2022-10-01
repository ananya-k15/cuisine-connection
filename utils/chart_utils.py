import itertools
import graphviz
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score, confusion_matrix


def generate_tree_file(bamboo_tree, cuisines, ingredients, filename):

    filename += ".dot"
    export_graphviz(
        bamboo_tree,
        feature_names=list(ingredients.columns.values),
        out_file=filename,
        class_names=np.unique(cuisines),
        filled=True,
        node_ids=True,
        special_characters=True,
        impurity=False,
        label="all",
        leaves_parallel=False,
    )


def display_tree(filename):
    filename += ".dot"
    with open(filename) as bamboo_tree_image:
        bamboo_tree_graph = bamboo_tree_image.read()
    return graphviz.Source(bamboo_tree_graph)


def generate_confusion_matrix(bamboo_test_cuisines, bamboo_pred_cuisines):
    test_cuisines = np.unique(bamboo_test_cuisines)
    bamboo_confusion_matrix = confusion_matrix(
        bamboo_test_cuisines, bamboo_pred_cuisines, test_cuisines
    )
    title = "Bamboo Confusion Matrix"
    cmap = plt.cm.Blues

    plt.figure(figsize=(8, 6))
    bamboo_confusion_matrix = (
        bamboo_confusion_matrix.astype("float")
        / bamboo_confusion_matrix.sum(axis=1)[:, np.newaxis]
    ) * 100

    plt.imshow(bamboo_confusion_matrix, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(test_cuisines))
    plt.xticks(tick_marks, test_cuisines)
    plt.yticks(tick_marks, test_cuisines)

    fmt = ".2f"
    thresh = bamboo_confusion_matrix.max() / 2.0
    for i, j in itertools.product(
        range(bamboo_confusion_matrix.shape[0]), range(bamboo_confusion_matrix.shape[1])
    ):
        plt.text(
            j,
            i,
            format(bamboo_confusion_matrix[i, j], fmt),
            horizontalalignment="center",
            color="white" if bamboo_confusion_matrix[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    return plt
