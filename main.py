import graphviz
import numpy as np
import pandas as pd
from sklearn import tree
from random import random
from sklearn.tree import export_graphviz

from utils.chart_utils import *
from utils.data_utils import *

# Get clean data
recipes = get_data()

# Select only asian cuisines
asian_recipes = recipes[
    recipes.cuisine.isin(["korean", "japanese", "chinese", "thai", "indian"])
]

# Set sample size for test set
sample_n = 30

# Generate test set
random.seed(1234)  # set random seed
asian_test = asian_recipes.groupby("cuisine", group_keys=False).apply(
    lambda x: x.sample(sample_n)
)
asian_test_ingredients = asian_test.iloc[:, 1:]
asian_test_cuisines = asian_test["cuisine"]

# Generate training set
asian_test_index = asian_recipes.index.isin(asian_test.index)
asian_train = asian_recipes[~asian_test_index]
asian_train_ingredients = asian_train.iloc[:, 1:]
asian_train_cuisines = asian_train["cuisine"]

# Create decision tree from training set
asian_train_tree = tree.DecisionTreeClassifier(max_depth=15)
asian_train_tree.fit(asian_train_ingredients, asian_train_cuisines)

# Visualize decision tree
export_graphviz(
    asian_train_tree,
    feature_names=list(asian_train_ingredients.columns.values),
    out_file="asian_train_tree.dot",
    class_names=np.unique(asian_train_cuisines),
    filled=True,
    node_ids=True,
    special_characters=True,
    impurity=False,
    label="all",
    leaves_parallel=False,
)
with open("asian_train_tree.dot") as asian_train_tree_image:
    asian_train_tree_graph = asian_train_tree_image.read()
graphviz.Source(asian_train_tree_graph)

# Predict cuisines from test set
asian_pred_cuisines = asian_train_tree.predict(asian_test_ingredients)

# Generate final confusion matrix
generate_confusion_matrix(asian_test_cuisines, asian_pred_cuisines)
