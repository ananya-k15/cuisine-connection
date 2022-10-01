import numpy as np
import pandas as pd
from sklearn import tree


def get_data():
    df = pd.read_csv("data/recipes.csv")
    return clean_df(df)


def clean_df(df):

    # fixing name of the column displaying the cuisine
    column_names = df.columns.values
    column_names[0] = "cuisine"
    df.columns = column_names

    # converting cuisine names to lower case
    df["cuisine"] = df["cuisine"].str.lower()

    # making the cuisine names consistent
    df.loc[df["cuisine"] == "austria", "cuisine"] = "austrian"
    df.loc[df["cuisine"] == "belgium", "cuisine"] = "belgian"
    df.loc[df["cuisine"] == "china", "cuisine"] = "chinese"
    df.loc[df["cuisine"] == "canada", "cuisine"] = "canadian"
    df.loc[df["cuisine"] == "netherlands", "cuisine"] = "dutch"
    df.loc[df["cuisine"] == "france", "cuisine"] = "french"
    df.loc[df["cuisine"] == "germany", "cuisine"] = "german"
    df.loc[df["cuisine"] == "india", "cuisine"] = "indian"
    df.loc[df["cuisine"] == "indonesia", "cuisine"] = "indonesian"
    df.loc[df["cuisine"] == "iran", "cuisine"] = "iranian"
    df.loc[df["cuisine"] == "italy", "cuisine"] = "italian"
    df.loc[df["cuisine"] == "japan", "cuisine"] = "japanese"
    df.loc[df["cuisine"] == "israel", "cuisine"] = "jewish"
    df.loc[df["cuisine"] == "korea", "cuisine"] = "korean"
    df.loc[df["cuisine"] == "lebanon", "cuisine"] = "lebanese"
    df.loc[df["cuisine"] == "malaysia", "cuisine"] = "malaysian"
    df.loc[df["cuisine"] == "mexico", "cuisine"] = "mexican"
    df.loc[df["cuisine"] == "pakistan", "cuisine"] = "pakistani"
    df.loc[df["cuisine"] == "philippines", "cuisine"] = "philippine"
    df.loc[df["cuisine"] == "scandinavia", "cuisine"] = "scandinavian"
    df.loc[df["cuisine"] == "spain", "cuisine"] = "spanish_portuguese"
    df.loc[df["cuisine"] == "portugal", "cuisine"] = "spanish_portuguese"
    df.loc[df["cuisine"] == "switzerland", "cuisine"] = "swiss"
    df.loc[df["cuisine"] == "thailand", "cuisine"] = "thai"
    df.loc[df["cuisine"] == "turkey", "cuisine"] = "turkish"
    df.loc[df["cuisine"] == "vietnam", "cuisine"] = "vietnamese"
    df.loc[df["cuisine"] == "uk-and-ireland", "cuisine"] = "uk-and-irish"
    df.loc[df["cuisine"] == "irish", "cuisine"] = "uk-and-irish"

    # removing data for cuisines with < 50 recipes:
    recipes_counts = df["cuisine"].value_counts()
    cuisines_indices = recipes_counts > 50

    cuisines_to_keep = list(
        np.array(recipes_counts.index.values)[np.array(cuisines_indices)]
    )
    df = df.loc[df["cuisine"].isin(cuisines_to_keep)]

    # converting all Yes's to 1's and the No's to 0's
    df = df.replace(to_replace="Yes", value=1)
    df = df.replace(to_replace="No", value=0)

    return df


def model_df(recipes, depth=3):

    # get list of cuisines and ingredients
    cuisines = recipes["cuisine"]
    ingredients = recipes.iloc[:, 1:]

    # create bamboo tree with specified depth
    bamboo_tree = tree.DecisionTreeClassifier(max_depth=depth)
    bamboo_tree = bamboo_tree.fit(ingredients, cuisines)

    return bamboo_tree
