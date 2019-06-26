import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


def likely_voters_series(voters_dict: dict) -> pd.Series:
    return pd.Series(list(voters_dict.values()), index=list(voters_dict.keys()))
    

class ElectionsResultsWrapper(BaseEstimator):
    """a wrapper for a model to the problem of predicting the distribution of votes between parties"""

    def __init__(self, model):
        super(ElectionsResultsWrapper, self).__init__()
        self.model = model
        self.targets = None

    def fit(self, x_train_set: pd.DataFrame, y_train_set: pd.DataFrame):
        """fits the model on a given train set and saves the labels"""
        self.model.fit(x_train_set, y_train_set)
        self.targets = y_train_set.unique()
        self.targets.sort()

    def predict(self, pred_set: pd.DataFrame):
        """
        predicts the distribution of votes between parties according to predict_proba
        which is a function that computes the probability of the model deciding to classify a voter to a certain party
        for every voter on the given dataset, and returns the sum of those probs for every party on all examples
        we then normalize by the sum the get the distribution
        """
        probs_predictions = self.model.predict_proba(pred_set)
        results = pd.Series(np.sum(probs_predictions, axis=0), index=self.targets)
        return results / np.sum(results.values)


class ElectionsWinnerWrapper(ElectionsResultsWrapper):
    """a wrapper for a model to the problem of predicting the party that wins the election"""

    def predict(self, pred_set: pd.DataFrame):
        """returns the prediction for the party to win the election
        according to the votes distrebution on the pred_set"""
        return self.predict_res(pred_set).idxmax()

    def predict_res(self, pred_set: pd.DataFrame):
        """
        calls to the function of ElectionsResultsWrapper that predicts the distrebution of votes between partys
        according to predict_proba
        """
        return super(ElectionsWinnerWrapper, self).predict(pred_set)


class LikelyVotersWrapper(BaseEstimator):
    """a wrapper for a model for the likely voters problem"""

    def __init__(self, model, threshold: float = 0.6):
        assert 0 < threshold < 1.0
        super(LikelyVotersWrapper, self).__init__()
        self.model = model
        self.threshold = threshold
        self.targets = None

    def fit(self, x_train_set: pd.DataFrame, y_train_set: pd.Series):
        """fits the wrapped model on the given dataset and saves the labels"""
        self.model.fit(x_train_set, y_train_set)
        self.targets = y_train_set.unique()
        self.targets.sort()

    def _get_party_likely_voters(self, df: pd.DataFrame, probs_predictions: pd.DataFrame, party: str):
        """
        return a list of indexes of voters who will likely vote for the given party
        decides witch voter will likely vote for the given party according to probs_prediction
        that contains the prob of every voter to vote to any of the partys , will list voters that their predicted
        prob to vote for the given party is above the threshold
        """
        return df.index[probs_predictions[party] > self.threshold]

    def predict(self, df: pd.DataFrame, party: str = None):
        """
        if a party is given returns the most likely voters to vote for the party
        if no party was given it will return the most likely voters to vote for every party
        predicts who is likely to vote for every party like what is explained
        in the description of _get_party_likely_voters
        """
        probs_predictions = pd.DataFrame(self.model.predict_proba(df), columns=self.targets)

        if party is not None:
            return self._get_party_likely_voters(df, probs_predictions, party)

        likely_voters = {tar: self._get_party_likely_voters(df, probs_predictions, tar) for tar in self.targets}
        used_indices = [list(indices) for indices in likely_voters.values()]
        likely_voters.update({None: df.index.difference(sum(used_indices, []))})

        return likely_voters_series(likely_voters)
