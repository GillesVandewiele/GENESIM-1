"""
0:[X10<3] yes=1,no=2,missing=1
	1:[X6<7] yes=3,no=4,missing=3
		3:[X5<9] yes=7,no=8,missing=7
			7:[X3<0.23] yes=13,no=14,missing=13
				13:[X5<7] yes=19,no=20,missing=19
					19:leaf=-0.100874
					20:leaf=0.0586768
				14:[X14<231] yes=21,no=22,missing=21
					21:leaf=-0.191359
					22:leaf=-0.0915265
			8:[X14<177] yes=15,no=16,missing=15
				15:[X3<4.375] yes=23,no=24,missing=23
					23:leaf=-0.134343
					24:leaf=-0
				16:[X14<1057] yes=25,no=26,missing=25
					25:leaf=0.118134
					26:leaf=0.0181373
		4:[X5<4] yes=9,no=10,missing=9
			9:leaf=-0.100874
			10:[X14<284] yes=17,no=18,missing=17
				17:[X14<2] yes=27,no=28,missing=27
					27:leaf=0.116584
					28:leaf=-0.0880152
				18:leaf=0.168124
	2:[X3<1.4375] yes=5,no=6,missing=5
		5:[X14<403] yes=11,no=12,missing=11
			11:leaf=-0.0470035
			12:leaf=0.141011
		6:leaf=0.197898



* \t geeft de huidige diepte van de boom aan
* [test]
* afhankelijk van resultaat van de test (yes no of missing), ga je naar de corresponderende node (bv yes=1,no=2,missing=1 --> ga naar 1 als yes of missing)
* Predictions are made by summing up the corresponding leaf values of each tree. Additionally, you need to transform those values depending on the objective you have choosen. For instance: If you trained your xgb with binary:logistic, the sum of the leaf values will be the logit score. So you need to apply the logistic function to get the wanted probabilities.
--> What happens with multi:logloss?????
"""
from sklearn.cross_validation import StratifiedKFold
from xgboost import plot_tree
import matplotlib.pyplot as plt

from constructors.ensemble import XGBClassification
from data.load_all_datasets import load_all_datasets
from sortedcontainers import SortedList, SortedSet

import numpy as np
from collections import Counter
import time

from decisiontree import DecisionTree


def parse_xgb_tree_string(tree_string, training_data, feature_cols, label_col, the_class):
    # Get class distribution
    _classes = np.unique(training_data[label_col].values)
    class_distributions = {}
    for _class in _classes:
        data = training_data[training_data[label_col] != _class]
        class_counts = Counter(data[label_col].values)
        total = sum(class_counts.values(), 0.0)
        for key in class_counts:
            class_counts[key] /= total
        class_distributions[_class] = class_counts

    # Get the unique values per feature
    unique_values_per_feature = {}
    for feature_col in feature_cols:
        # Just use a simple sorted list of np.unique (faster than SortedList or a set to get index of elements after testing)
        unique_values_per_feature[feature_col] = sorted(np.unique(training_data[feature_col].values))

    return parse_xgb_tree(tree_string, _class=the_class, class_distributions=class_distributions,
                          unique_values_per_feature=unique_values_per_feature, n_samples=len(training_data))


def get_closest_value(x, values):
    for i in range(len(values)-1):
        if values[i+1] > x:
            return float(values[i])
    return x


def parse_xgb_tree(tree_string, _class=0, class_distributions={}, unique_values_per_feature={}, n_samples=0):
    # There is some magic involved! The leaf values need to be converted to class distributions somehow!
    # For binary classification problems: convert to probability by calculating 1/(1+exp(-value))
    # For multi_class: the tree_string contains n_estimators * n_classes decision trees
    # WARNING: Classes are sorted according to output of np.unique
    # Ordered as follows [tree_1-class_1, ..., tree_1-class_k, tree_2-class_1, ....]

    # The problem is: tree_i is different for each class...
    # One possibility is to assign the probability to that class by calculating logistic function
    # And dividing the rest of the probability (sum to 1) according to the distribution of the remaining classes

    # Next problem is: everything is expressed as "feature < threshold" instead "feature <= threshold"
    # Solution is: take the infimum of those feature values

    decision_trees = {}
    # Binary classification
    binary_classification = len(class_distributions.keys()) == 2
    for line in tree_string.split('\n'):
        if line != '':
            _id, rest = line.split(':')
            _id = _id.lstrip()
            if rest[:4] != 'leaf':
                feature = rest.split('<')[0][1:]
                highest_lower_threshold = get_closest_value(float(rest.split('<')[1].split(']')[0]),
                                                            unique_values_per_feature[feature])
                decision_trees[_id] = DecisionTree(right=None, left=None,
                                                   label=feature, value=highest_lower_threshold,
                                                   parent=None)
            else:
                leaf_value = float(rest.split('=')[1])
                if binary_classification:
                    probability = 1/(1+np.exp(leaf_value))
                    other_class = class_distributions[_class].keys()[0]
                    class_probs = {_class: int(n_samples*probability),
                                   other_class: int(n_samples*(1-probability))}
                    if probability > 0.5: most_probable_class = _class
                    else: most_probable_class = other_class
                else:
                    probability = 1/(1+np.exp(-leaf_value))
                    class_probs = {}
                    remainder_samples = int(n_samples - probability*n_samples)
                    class_probs[_class] = int(probability*n_samples)
                    most_probable_class, most_samples = _class, class_probs[_class]
                    for other_class in class_distributions[_class]:
                        amount_samples = int(remainder_samples*class_distributions[_class][other_class])
                        class_probs[other_class] = amount_samples
                        if amount_samples > most_samples:
                            most_probable_class = other_class
                            most_samples = amount_samples

                decision_trees[_id] = DecisionTree(right=None, left=None,
                                                   label=most_probable_class, value=None,
                                                   parent=None)
                decision_trees[_id].class_probabilities = class_probs

    # Make another pass to link the different decision trees together
    for line in tree_string.split('\n'):
        if line != '':
            _id, rest = line.split(':')
            _id = _id.lstrip()
            tree = decision_trees[_id]
            if rest[:4] != 'leaf':
                rest = rest.split(']')[1].lstrip()
                links = rest.split(',')
                for link in links:
                    word, link_id = link.split('=')
                    if word == 'yes' or word == 'missing':
                        tree.left = decision_trees[link_id]
                    else:
                        tree.right = decision_trees[link_id]

    return decision_trees['0']

NR_FOLDS = 5
xgb = XGBClassification()
for dataset in load_all_datasets():
    df = dataset['dataframe']
    label_col = dataset['label_col']
    feature_cols = dataset['feature_cols']

    skf = StratifiedKFold(df[label_col], n_folds=NR_FOLDS, shuffle=True, random_state=1337)

    for fold, (train_idx, test_idx) in enumerate(skf):
        print 'Fold', fold + 1, '/', NR_FOLDS, 'for dataset', dataset['name']
        train = df.iloc[train_idx, :].reset_index(drop=True)
        X_train = train.drop(label_col, axis=1)
        y_train = train[label_col]
        test = df.iloc[test_idx, :].reset_index(drop=True)
        X_test = test.drop(label_col, axis=1)
        y_test = test[label_col]

        xgb_model = xgb.construct_classifier(train, feature_cols, label_col)

        n_classes = len(np.unique(y_train.values))
        if n_classes > 2:
            for idx, tree_string in enumerate(xgb_model.clf._Booster.get_dump()):
                tree = parse_xgb_tree_string(tree_string, train, feature_cols, label_col, np.unique(y_train.values)[idx % n_classes])
                tree.visualise('xgbtree')
                plot_tree(xgb_model.clf, num_trees=idx)
                plt.show()
                raw_input()
        else:
            for idx, tree_string in enumerate(xgb_model.clf._Booster.get_dump()):
                tree = parse_xgb_tree_string(tree_string, train, feature_cols, label_col, 0)
                tree.visualise('xgbtree')
                plot_tree(xgb_model.clf, num_trees=idx)
                plt.show()
                raw_input()

