import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.utils import resample
from sklearn.metrics import f1_score, balanced_accuracy_score
from numpy.linalg import norm
from hyper_params_funcs import *


def evaluate_voters_division(estimator, X, y_true) -> float:
    y_pred = estimator.predict(X)
    return balanced_accuracy_score(y_true, y_pred)


def hist_softmax(hist_in: pd.Series, T: float = 1.0):
    if T is None:
        T = len(hist_in.index)

    hist = np.exp(hist_in.astype(float) / T)
    hist = hist / np.sum(hist)

    return hist


def evaluate_election_winner(estimator, X, y_true) -> float:
    hist_true = hist_softmax(y_true.value_counts().astype(float) / len(y_true.index))
    hist_pred = hist_softmax(estimator.predict_res(X))

    return -norm(hist_true - hist_pred)


def evaluate_election_res(estimator, X, y_true) -> float:
    hist_true = y_true.value_counts().astype(float) / len(y_true.index)
    hist_pred = estimator.predict(X)

    return -norm(hist_true - hist_pred)


def evaluate_party_voters(indices_pred, y_true: pd.Series):
    """
    evaluate the likely voters problem on a specific party
    :param indices_pred:
    :param y_true:
    :return:
    """
    y_pred = y_true.copy()
    y_pred[indices_pred] = True
    y_pred[y_pred.index.difference(indices_pred)] = False

    # in the case that the recall and accuracy are both 0, return the minimal value of 0
    # this happens when the intersection is empty, which can be seen if the two have no shared True values
    # and they are not both all False values
    if 0 == len(y_true.index[y_true].intersection(indices_pred)) < max(len(y_true.index[y_true]), len(indices_pred)):
        return 0.0

    return f1_score(y_true, y_pred)


def evaluate_likely_voters(estimator, X, y_true: pd.Series):
    """
    evaluate the prediction of the model on the likely voters problem
    :param estimator: estimator
    :param y_true: should be valid['Vote'] or something like that
    :param X: should be valid[features] or something like that
    :return: average score for each party
    """
    parties = estimator.targets
    y_pred = estimator.predict(X)

    return np.mean([evaluate_party_voters(y_pred[party], y_true == party) for party in parties])


def upsample(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    balances a dataframe according to a target label
    :param df: the dataframe
    :param target: the target label
    :return: the dataframe, now balanced according to the target values
    """
    targets = df[target]
    classes = targets.unique()

    # number of appearances per class
    num_appearances = {target_class: targets[targets == target_class].size for target_class in classes}
    # basically argmax
    most_common = max(num_appearances.iterkeys(), key=(lambda key: num_appearances[key]))

    # upsampled sub-dataframes for the values that aren't the most common
    minority_upsampled = [resample(df[targets == target_class], replace=True, n_samples=num_appearances[most_common])
                          for target_class in classes if target_class != most_common]

    # balanced dataframe
    df_upsampled = pd.concat([df[targets == most_common]] + minority_upsampled)

    return df_upsampled


def target_features_split(df: pd.DataFrame, target: str):
    """
    splits the dataframe to features and target label (X and y)
    :param df: the dataframe
    :param target: the target label
    :return: 2 dataframes, the first with only the features, the second with only the labels
    """
    features = list(set(df.columns.values.tolist()).difference({target}))
    return df[features], df[target]


def cross_valid(model, df: pd.DataFrame, num_folds: int, eval_func, target='Vote'):
    """
    peforms k-fold cross validation on the dataframe df according to the evaluation function
    :param model: the model we are checking
    :param df: the dataframe we want to cross validate
    :param num_folds: the number of folds
    :param eval_func: the evaluation function
    :param target: the target label
    :return: the average score across the folds
    """
    kf = StratifiedKFold(n_splits=num_folds)
    score = 0

    df_features, df_targets = target_features_split(df, target)

    for train_indices, test_indices in kf.split(df_features, df_targets):
        train_targets, train_features = df_targets[train_indices], df_features.iloc[train_indices]
        test_targets, test_features = df_targets[test_indices], df_features.iloc[test_indices]

        model.fit(train_features, train_targets)
        score += eval_func(model, test_features, test_targets)

    return score / num_folds


def choose_best_model(models, valid: pd.DataFrame, eval_func, verbose: bool = False):
    """
    return the model that performs the best on the validation set according to the evaluation function
    :param models: list of models to check, already fitterd on the training set
    :param valid: the validation set
    :param eval_func: the evaluation function for the problem
    :param verbose: verbose flag
    :return: the best performing estimator
    """
    best_score = -np.inf
    best_model = None

    # train_features, train_targets = target_features_split(train, 'Vote')
    valid_features, valid_targets = target_features_split(valid, 'Vote')

    for model in models:
        # model.fit(train_features, train_targets)
        score = eval_func(model, valid_features, valid_targets)

        if verbose:
            print(f'Model {get_model_name(model)} has a score of {score}')

        if score > best_score:
            best_score = score
            best_model = model

    return best_model


def wrapper_params(params: dict):
    """
    converts the inputted params dictionary into ones that fit with the wrappers
    :param params: parameters dictionary (each key is a hyper-paramter)
    :return: converted params to fit the wrappers
    """
    return {'model__' + key: value for key, value in params.items()}


def is_model_balanced(model) -> bool:
    """
    :param model: the model that checked
    :return: True if the model balances the class weight of the labels, False otherwise
    """
    params = get_normal_params(model.get_params())
    return 'class_weight' in params.keys() and params['class_weight'] == 'balanced'


def model_train_set(model, train, target):
    """
    function that returns a balanced training set if the model does not balance on it's own
    :param model: the model
    :param train: the actual training set
    :param target: the target label
    :return: if the model already balances on it's own - train, otherwise a balanced version of train
    """
    return train if is_model_balanced(model) else upsample(train, target)


def choose_hyper_params(models, params_ranges, eval_func, train, target='Vote', num_folds=3, wrapper=None,
                        random_state=None, n_iter=10, verbose=False):
    """
    given a list of models and an evaluation function, find the best hyper parameters for each model to get the best score
    :param models: list of models
    :param params_ranges: parameter distributions for each estimator
    :param eval_func: evaluation function for the problem
    :param train: training set
    :param target: the target label
    :param num_folds: the number of folds used in cross-validation
    :param wrapper: the wrapper class used for the models on the current problem (None if there isn't one)
    :param random_state: random state (int or np.RandomState) to use for the randomized search
    :param n_iter: number of iterations for the randomized search on each estimator
    :param verbose: verbose flag
    :return: the estimators with tuned hyper parameters fitted on the training set
    """

    used_models = models
    if wrapper is not None:
        used_models = [wrapper(model) for model in models]

    best_models = []
    for model, params in zip(used_models, params_ranges):
        if verbose:
            print(f'Tuning model #{len(best_models) + 1}: {get_model_name(model)}')

        used_params = params
        if wrapper is not None:
            used_params = wrapper_params(params)

        grid = RandomizedSearchCV(model, used_params, scoring=eval_func, cv=num_folds, random_state=random_state,
                                  n_iter=n_iter, n_jobs=-1)

        grid.fit(*target_features_split(model_train_set(model, train, target), target))
        best_models.append(grid.best_estimator_)

    return best_models


def find_problem_best_model(train, valid, estimators, params, problem, eval_func, wrapper, n_iter=10, seed=None,
                            search_hyper_params=True, verbose=False):
    """
    :param train: train set
    :param valid: validiation set
    :param estimators: estimators that we should
    :param params: parameter distributions for each estimator
    :param problem: the name of the current problem
    :param eval_func: evaluation function for said problem
    :param wrapper: the wrapper for the problem
    :param n_iter: number of random guesses per estimator
    :param seed: random seed for random search
    :param search_hyper_params: boolean flag for whether we want to do a random search or load from file
    :param verbose: flag for verbose prints
    :return: the best performing estimator for the problem
    """
    print('============================================')
    print(f'started {problem}')
    if search_hyper_params:
        best_estimators = choose_hyper_params(estimators, params, eval_func, train, 'Vote', random_state=seed,
                                              n_iter=n_iter, wrapper=wrapper, verbose=verbose)
        save_problem_hyper_params(best_estimators, problem)
    else:
        best_estimators = load_problem_hyper_params(estimators, problem, verbose=verbose, wrapper=wrapper)
        for estimator in best_estimators:
            estimator.fit(*target_features_split(model_train_set(estimator, train, 'Vote'), 'Vote'))

    print_best_hyper_params(best_estimators, problem)
    best_estimator = choose_best_model(best_estimators, valid, eval_func, verbose=verbose)
    print_best_model(best_estimator, problem)

    return best_estimator
