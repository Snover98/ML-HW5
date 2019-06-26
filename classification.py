import pandas as pd
import sklearn as sk
from data_preperation import prepare_data
from wrappers import *
from model_selection import *
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from pprint import pprint
from argparse import ArgumentParser


class LogUniform:
    def __init__(self, low=0.0, high=1.0, size=None, base=10):
        self.low = low
        self.high = high
        self.size = size
        self.base = base

    def rvs(self, random_state):
        if random_state is None:
            state = np.random
        elif isinstance(random_state, np.random.RandomState):
            state = random_state
        else:
            state = np.random.RandomState(random_state)

        return np.power(self.base, state.uniform(self.low, self.high, self.size))


class RandIntMult:
    def __init__(self, low=0.0, high=1.0, mult=1, size=None):
        self.low = low
        self.high = high
        self.size = size
        self.mult = mult

    def rvs(self, random_state):
        if random_state is None:
            state = np.random
        elif isinstance(random_state, np.random.RandomState):
            state = random_state
        else:
            state = np.random.RandomState(random_state)

        return np.around(state.uniform(low=self.low, high=self.high, size=self.size) * self.mult).astype(int)


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    return ax


def print_likely_voters(likely_voters):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        for index, value in likely_voters.iteritems():
            print(f'{index}: {list(value)}')


def find_best_models(train, valid, search_hyper_params=True, verbose=False):
    seed = np.random.randint(2 ** 31)
    print(f'seed is {seed}')
    print('')

    n_iter = 10

    random_forest_params = {
        'n_estimators': RandIntMult(low=0.5, high=20.0, mult=100),
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2', None],
        'bootstrap': [True, False],
        'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]
    }

    svc_params = {
        'C': LogUniform(low=-5.0, high=4.0),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [3, 4, 5],
        'gamma': ['scale', 'auto'],
        'tol': LogUniform(low=-10, high=0),
        'coef0': [0.0, 1.0]
    }

    estimators = [SVC(probability=True, class_weight='balanced'), RandomForestClassifier(class_weight='balanced')]
    params = [svc_params, random_forest_params]

    problems = ['voter classification', 'election winner', 'election results', 'likely voters']
    wrappers = [None, ElectionsWinnerWrapper, ElectionsResultsWrapper, LikelyVotersWrapper]
    eval_funcs = [evaluate_voters_division, evaluate_election_winner, evaluate_election_res, evaluate_likely_voters]

    best_estimators = [
        find_problem_best_model(train, valid, estimators, params, problem, eval_func, wrapper, n_iter, seed,
                                search_hyper_params, verbose) for problem, eval_func, wrapper in
        zip(problems, eval_funcs, wrappers)]

    return best_estimators


def use_estimators(best_estimators, train, valid, test):
    best_normal, best_election_win, best_election_res, best_likely_voters_model = best_estimators

    features = list(set(train.columns.to_numpy().tolist()).difference({'Vote'}))

    non_test_data = pd.concat((train, valid))

    # data for the final classifier
    print('')
    print('============================================')
    best_normal.fit(*target_features_split(non_test_data, 'Vote'))
    test_pred = pd.Series(best_normal.predict(test[features]), index=test.index)
    test_true = test['Vote']
    parties = non_test_data['Vote'].unique()
    plot_confusion_matrix(test_true, test_pred, parties)
    print('')
    test_pred.to_csv('test_predictions.csv', index=False)

    # predict elections winner
    print('============================================')
    best_election_win.fit(*target_features_split(non_test_data, 'Vote'))
    pred_election_winner = best_election_win.predict(test[features])
    true_election_winner = test['Vote'].value_counts().idxmax()
    print(f'The predicted elections winner is {pred_election_winner} and the actual winner is {true_election_winner}')
    print('')

    # predict elections results
    print('============================================')
    best_election_res.fit(*target_features_split(non_test_data, 'Vote'))
    pred_percentages = best_election_res.predict(train[features]) * 100
    true_percentages = test['Vote'].value_counts() / len(test.index) * 100
    print('The predicted distribution of votes across the parties is:')
    pprint(pred_percentages)
    print('The true distribution of votes across the parties is:')
    pprint(true_percentages[pred_percentages.index])
    print('')

    # predict likely voters
    print('============================================')
    best_likely_voters_model.fit(*target_features_split(non_test_data, 'Vote'))
    pred_likely_voters = best_likely_voters_model.predict(test[features])
    actual_voters = likely_voters_series(
        {party: test['Vote'].index[test['Vote'] == party] for party in non_test_data['Vote'].unique()})
    print('Predicted likely voter indices per party:')
    print_likely_voters(pred_likely_voters)
    print('')
    print('Actual voter indices per party:')
    print_likely_voters(actual_voters)
    print('')


def main():
    parser = ArgumentParser()
    parser.add_argument('--verbose', '-v', action='store_true', dest='verbose', default=False)
    parser.add_argument('--search_hyper_params', '-s', action='store_true', dest='search_hyper_params', default=False)
    parser.add_argument('--preprocess_data', '-p', action='store_true', dest='preprocess_data', default=False)

    args = vars(parser.parse_args())

    if args['preprocess_data']:
        train, valid, test = prepare_data()
    else:
        train = pd.read_csv('train_processed.csv')
        valid = pd.read_csv('valid_processed.csv')
        test = pd.read_csv('test_processed.csv')

    verbose = args['verbose']
    search_hyper_params = args['search_hyper_params']

    best_models = find_best_models(train, valid, verbose=verbose, search_hyper_params=search_hyper_params)
    use_estimators(best_models, train, valid, test)


if __name__ == '__main__':
    main()
