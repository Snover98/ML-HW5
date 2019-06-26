import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from standartisation import DFScaler
from wrappers import *
from model_selection import target_features_split
from sklearn.svm import SVC
import pickle


def check_interval(scaler: DFScaler, changed_data: pd.DataFrame, low: int, high: int, model, feature: str,
                   hist_true: pd.Series, num_iters=50):
    print('===============================================')
    print(f'interval is [{low}, {high})')
    hists = pd.Series(np.zeros_like(hist_true.values), index=hist_true.index)
    for idx in range(num_iters):
        data_cp = changed_data.copy()
        data_cp[feature] = pd.Series(np.random.uniform(low, high, size=len(changed_data)), index=changed_data.index)

        scaled_changed_data = scaler.scale(data_cp)
        scaled_changed_X, _ = target_features_split(scaled_changed_data, "Vote")

        hists += model.predict(scaled_changed_X)
        if idx % 10 == 0:
            print('!', end='')
        else:
            print('.', end='')

    print('!')

    hist = hists / num_iters

    parties = hist.index
    print(hist_true[parties] * 100)
    print(hist[parties] * 100)
    print((hist[parties] - hist_true[parties]) * 100)
    print('')


def main():
    fitted = True
    data = pd.read_csv("full_train.csv")
    test = pd.read_csv("full_test.csv")

    test_X, test_y = target_features_split(test, "Vote")

    features = test.columns.values.tolist()
    selected_features = ["Avg_environmental_importance", "Avg_government_satisfaction", "Avg_education_importance",
                         "Avg_monthly_expense_on_pets_or_plants", "Avg_Residancy_Altitude", "Yearly_ExpensesK",
                         "Weighted_education_rank", "Number_of_valued_Kneset_members"]
    issues = [feat for feat in features if feat.startswith("Issue")]
    features = issues + selected_features + ["Vote"]

    data = data[features]
    test = test[features]

    changed_data = data.copy()

    scaler = DFScaler(data, selected_features)
    data = scaler.scale(data)
    test = scaler.scale(test)

    X, Y = target_features_split(data, "Vote")
    hist_true = Y.value_counts().astype(float) / len(Y.index)

    if not fitted:
        model = ElectionsResultsWrapper(
            SVC(C=7.70625, class_weight='balanced', degree=5, gamma='auto', kernel='poly', probability=True,
                tol=0.33618))
        model.fit(X, Y)
        pickle.dump(model, open('fit_model.sav', 'wb'))
    else:
        model = pickle.load(open('fit_model.sav', 'rb'))

    feature = "Avg_Residancy_Altitude"

    intervals = [0, 2, 4, 6, 10, 12, 13]

    print(f'feature is {feature}')
    for low, high in zip(intervals[:-1], intervals[1:]):
        check_interval(scaler, changed_data, low, high, model, feature, hist_true)


if __name__ == "__main__":
    main()
