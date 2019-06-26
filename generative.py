import sklearn
import pandas as pd
import numpy as np
from model_selection import target_features_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import davies_bouldin_score


# Distribution = namedtuple('Distribution',)


def train_generative(data: pd.DataFrame):
    models = [LinearDiscriminantAnalysis(store_covariance=True), QuadraticDiscriminantAnalysis(store_covariance=True)]
    X, Y = target_features_split(data, "Vote")
    for model in models:
        model.fit(X, Y)
    return models


# def KL(classifier, p_idx, q_idx, size=1024, is_lda=False):
#     if is_lda:
#         p_mean = classifier.means_[p_idx]
#         p_cov = classifier.covariance_
#     else:
#         p_mean = classifier.means_[p_idx]
#         p_cov = classifier.covariance_[p_idx]
#
#     samples = np.random.multivariate_normal(p_mean, p_cov, size=size)
#
#     log_probs = np.mean(classifier.predict_log_proba(samples), axis=0)
#     log_p, log_q = log_probs[:, p_idx], log_probs[:, q_idx]
#
#     return log_p - log_q


def jensen_shannon(classifier, p_idx, q_idx, size=4096, is_lda=False):
    if is_lda:
        p_mean = classifier.means_[p_idx]
        p_cov = classifier.covariance_

        q_mean = classifier.means_[q_idx]
        q_cov = classifier.covariance_
    else:
        p_mean = classifier.means_[p_idx]
        p_cov = classifier.covariance_[p_idx]

        q_mean = classifier.means_[q_idx]
        q_cov = classifier.covariance_[q_idx]

    samples_p = np.random.multivariate_normal(p_mean, p_cov, size=size)

    log_probs_p = classifier.predict_log_proba(samples_p)
    log_p_true, log_q_false = log_probs_p[:, p_idx], log_probs_p[:, q_idx]
    log_mix_p = np.logaddexp(log_p_true, log_q_false)

    samples_q = np.random.multivariate_normal(q_mean, q_cov, size=size)

    log_probs_q = classifier.predict_log_proba(samples_q)
    log_p_false, log_q_true = log_probs_q[:, p_idx], log_probs_p[:, q_idx]
    log_mix_q = np.logaddexp(log_p_false, log_q_true)

    return (log_p_true.mean() - (log_mix_p.mean() - np.log(2)) + log_q_true.mean() - (log_mix_q.mean() - np.log(2))) / 2


def is_coalition_ready(features: pd.DataFrame, labels: pd.Series, groups, next_groups):
    if next_groups is None or groups is None:
        return False

    cur_col = get_coalition(labels, groups)
    next_col = get_coalition(labels, next_groups)

    if cur_col is None or next_col is None:
        return False
    elif len(set(next_col).difference(labels.unique())) == 0:
        return True

    cur_score = davies_bouldin_score(features, labels.isin(cur_col).astype(np.int))
    next_score = davies_bouldin_score(features, labels.isin(next_col).astype(np.int))

    return cur_score >= next_score


def get_coalition(labels: pd.Series, groups):
    num_voters = labels.to_numpy().shape[0]

    for group in groups:
        if labels[labels.isin(group)].to_numpy().shape[0] > num_voters / 2.0:
            return group

    return None


def create_gen_coalition(gen_model, features: pd.DataFrame, labels: pd.Series):
    n_labels = len(labels.unique())

    js_matrice = np.zeros((n_labels, n_labels))

    is_lda = isinstance(gen_model, LinearDiscriminantAnalysis)

    for idx_1 in range(len(labels.unique())):
        for idx_2 in range(idx_1):
            js_matrice[idx_1, idx_2] = jensen_shannon(gen_model, idx_1, idx_2, is_lda=is_lda)
            js_matrice[idx_2, idx_1] = js_matrice[idx_1, idx_2]

    next_groups = [[label] for label in labels.unique()]
    groups = None
    while not is_coalition_ready(features, labels, groups, next_groups):
        groups = next_groups
        next_groups = unite_parties(labels, js_matrice, groups)

    coalition = get_coalition(labels, groups)

    return coalition


def unite_parties(labels: pd.Series, js_matrice: np.ndarray, groups):
    parties = list(labels.unique())

    dists = np.zeros((len(groups), len(groups)))

    for group_idx, group in enumerate(groups):
        group_indices = [parties.index(party) for party in group]
        dists[group_idx, group_idx] = np.inf
        for other_group_idx in range(group_idx):
            other_group_indices = [parties.index(party) for party in groups[other_group_idx]]
            dists[group_idx, other_group_idx] = np.mean(js_matrice[group_indices, :][:, other_group_indices], axis=None)
            dists[other_group_idx, group_idx] = dists[group_idx, other_group_idx]

    merge_idx1, merge_idx2 = np.unravel_index(dists.argmin(), dists.shape)
    merged = groups[merge_idx1] + groups[merge_idx2]

    return [groups[idx] for idx in range(len(groups)) if idx != merge_idx1 and idx != merge_idx2] + [merged]


def create_gen_coalitions(gen_models, features: pd.DataFrame, labels: pd.Series):
    return [create_gen_coalition(model, features, labels) for model in gen_models]
