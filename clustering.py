import sklearn as sk
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from itertools import cycle, islice
import matplotlib.colors as mcolors
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import BayesianGaussianMixture
from mpl_toolkits.mplot3d import Axes3D


def create_cluster_coalitions(models: list, data: pd.DataFrame, labels: pd.Series, threshold: float = 0.3):
    coalitions = []
    for model in models:
        probs = model.predict_proba(data)
        probs[probs < threshold] = 0
        con_mat = probs @ probs.transpose()
        col = sk.cluster.AgglomerativeClustering(n_clusters=2, connectivity=con_mat).fit_predict(data)

        col = pd.Series(col, index=labels.index)
        col_number = col.value_counts().idxmax()
        # parties = pd.Series()
        col_parties = labels.unique()[
            labels[col == col_number].value_counts().sort_index() >= labels.value_counts().sort_index() * 0.75]

        coalitions.append(col_parties)
    return coalitions


def get_clustering(features: pd.DataFrame, labels: pd.Series):
    n_clusters = 3
    clustering_methods = [MiniBatchKMeans(n_clusters=n_clusters), BayesianGaussianMixture(n_components=n_clusters)]
    return learn_clusters(features, labels, clustering_methods)


def cluster_party(features_subset: pd.DataFrame, party: str, clustering_method):
    clusters = clustering_method.fit_predict(features_subset)
    return party + pd.Series(clusters, index=features_subset.index).astype(str)


def cluster_parties(features: pd.DataFrame, labels: pd.Series, clustering_method):
    parties_clusters = [cluster_party(features[labels == party], party, clustering_method) for party in labels.unique()]
    return pd.concat(parties_clusters).sort_index()


def learn_clusters(features: pd.DataFrame, labels: pd.Series, clustering_methods):
    clustering_classifiers = []
    params = {
        'bootstrap': False,
        'class_weight': 'balanced',
        'criterion': 'gini',
        'max_depth': 20,
        'max_features': 'log2',
        'min_samples_leaf': 4,
        'min_samples_split': 10,
        'n_estimators': 1738,
        'warm_start': False
    }

    for method in clustering_methods:
        clf = RandomForestClassifier(n_jobs=-1)
        clf.set_params(**params)

        clusters = cluster_parties(features, labels, method)

        clf.fit(features, clusters)
        clustering_classifiers.append(clf)

    return clustering_classifiers


def show_clusters(features: pd.DataFrame, clustering_clf, title: str):
    y_pred = pd.Series(clustering_clf.predict(features)).astype('category').cat.codes
    show_labels(features, y_pred, title)


def show_labels(features: pd.DataFrame, y_pred: pd.Series, title: str):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = np.append(prop_cycle.by_key()['color'], [])

    if len(colors) < len(y_pred.unique()):
        added_colors = np.random.choice(list(mcolors.XKCD_COLORS.values()), size=len(y_pred.unique()) - len(colors))
        colors = np.append(colors, added_colors)

    pca = PCA(n_components=3, svd_solver='randomized')
    X = pca.fit(features).transform(features)

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_title(title)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=10, color=colors[y_pred.values])

    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.xticks(())
    plt.yticks(())
    plt.show()
