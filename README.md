# Cuisine Connection

This repository trains a supervised machine learning model to predict the country of origin of a recipe on the basis of the ingredients used in it using a decision tree. We start by training the data on a subset

---

## Table of Contents

1. [About the Data](#0)<br>
2. [Preprocessing](#1)<br>
3. [Data Modeling](#2)<br>
4. [Model Evaluation](#4)<br>

<hr>

## About the data <a id="0"></a>

In 2011, Yong-Yeol Ahn, Sebastian E. Ahnert, James P. Bagrow and Albert-László Barabási published a research paper on _Flavor network and the principles of food pairing_. The paper explored the existence of general patterns that determine the ingredient combinations used in food today or principles that transcend individual tastes and recipes. During their research, they scraped tens of thousands of food recipes (cuisines and ingredients) from three different websites, namely:

|                                                             All Recipes                                                             |                                                            Epic Curious                                                             |                                                             Menu Pan                                                             |
| :---------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------: |
| ![](https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DS0103EN/labs/images/lab4_fig1_allrecipes.png) | ![](https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DS0103EN/labs/images/lab4_fig2_epicurious.png) | ![](https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DS0103EN/labs/images/lab4_fig3_menupan.png) |
|                                                         www.allrecipes.com                                                          |                                                         www.epicurious.com                                                          |                                                         www.menupan.com                                                          |

Read the research summary at [Flavor Network and the Principles of Food Pairing](http://yongyeol.com/papers/ahn-flavornet-2011.pdf).

## Preprocessing <a id="1"></a>

Since the researchers have already processed and compiled the dataset, it is fairly reliable. However, we will take the following steps to clean the dataset :

1. Fix column names
2. Change the cuisine names for better readability
3. Remove data for cuisines with less than < 50 recipes to avoid a convoluted decision tree
4. Convert all the Yes's and No's to 1's and 0's for easy processing

Now, we can use this data to build a supervised decision tree which given a set of recipe ingredients, will predict the recipe's cuisine.

## Data Modeling <a id="2"></a>

### Removing dataset bias

A preliminary examination of the dataset shows that while we have data on a multitude of cuisines, there is a clear bias towards American recipes. To ensure that our decision tree is not biased towards American cuisine, we can either exclude American recipes from the dataset or build decision trees for different subsets of the data. In this project, we will adopt the latter solution, i.e., build a decision tree based on Asian cuisines.

### Setting tree depth

After some trial and error, a tree depth of three was found to be optimal for the decision tree.
