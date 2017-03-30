import warnings

from scipy.interpolate import splrep, splev
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import make_scorer

from data.load_all_datasets import load_all_datasets

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve


def less_strict_accuracy(y, y_pred, **kwargs):
    return float(sum((abs(list(y)[i]-list(y_pred)[i]) <= 1) for i in range(len(y))))/float(len(y))


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        scoring=make_scorer(less_strict_accuracy))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # pol1 = splrep(train_sizes, train_scores_mean, w=1./train_scores_std)
    # pol2 = splrep(train_sizes, test_scores_mean, w=1./test_scores_std)
    # print 'train:', train_scores_mean
    # print 'test:', test_scores_mean
    # print 'train_std:', train_scores_std
    # print 'test_std:', test_scores_std
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

NR_FOLDS = 5
NR_FEATURES = 40

for dataset in load_all_datasets():
    df = dataset['dataframe']
    label_col = dataset['label_col']
    feature_cols = dataset['feature_cols']
    train_sizes = np.linspace(0.05,1.0,num=20)
    skf = StratifiedKFold(df[label_col].values, n_folds=NR_FOLDS, shuffle=True, random_state=1337)
    params = {'C':[1,5,10,0.1,0.01], 'solver': ['newton-cg', 'lbfgs', 'liblinear']}
    lda = LinearDiscriminantAnalysis()
    plot_learning_curve(lda, 'learning curve for all features (LR 5-fold CV)', df[feature_cols], df[label_col], train_sizes=train_sizes, cv=skf)
    plt.show()