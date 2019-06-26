import sklearn
import pandas as pd
import numpy as np
import cleansing
from imputation import *
from standartisation import *
from wrappers import *
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from model_selection import target_features_split


def show_col_parties(coalition: pd.Series, labels: pd.Series):
    col_number = coalition.value_counts().idxmax()
    print('The coalition voters are:')
    print(labels[coalition == col_number].value_counts().sort_index())
    print('Out of:')
    print(labels.value_counts().sort_index())
    print(f'The coalition parties are {labels[coalition == col_number].unique()}')
    print(f'The Coalition has {100 * len(labels[coalition == col_number]) / len(coalition)}% of the votes')


def main():
    train = pd.concat([pd.read_csv('train_processed.csv'), pd.read_csv('test_processed.csv')], ignore_index=True)
    valid = pd.read_csv('valid_processed.csv')

    full_train = pd.concat([pd.read_csv('full_train.csv'), pd.read_csv('full_test.csv')], ignore_index=True)

    features = train.columns.tolist()
    selected_features = ["Avg_environmental_importance", "Avg_government_satisfaction", "Avg_education_importance",
                         "Avg_monthly_expense_on_pets_or_plants", "Avg_Residancy_Altitude", "Yearly_ExpensesK",
                         "Weighted_education_rank", "Number_of_valued_Kneset_members"]
    features = [feat for feat in features if feat.startswith("Issue")] + selected_features
    test = pd.read_csv("ElectionsData_Pred_Features.csv")
    test = pd.DataFrame(cleansing.cleanse(test))
    id_Num = test["IdentityCard_Num"]
    new_imputation(test, full_train)
    scaler = DFScaler(test, selected_features)
    test = scaler.scale(test)
    test = test[features]

    x_train, y_train = target_features_split(train, "Vote")

    svc = SVC(C=7.70625, class_weight='balanced', degree=5, gamma='auto', kernel='poly', probability=True, tol=0.33618)
    model = ElectionsResultsWrapper(svc)
    model.fit(x_train, y_train)

    election_results = model.predict(test)
    election_winner = election_results.idxmax()
    print(election_results)
    print("the winner:", election_winner)

    clf = RandomForestClassifier(bootstrap=False, min_samples_split=10, min_samples_leaf=4, criterion='gini',
                                 max_depth=20, max_features='log2', n_estimators=1738, class_weight='balanced')
    clf.fit(x_train, y_train)
    votes_predictions = clf.predict(test)
    results = pd.DataFrame()
    results["IdentityCard_Num"] = id_Num
    results["PredictVote"] = votes_predictions

    results.to_csv("election_results.csv", index=False)
    test["Vote"] = results["PredictVote"]
    test.to_csv("new_test_processed.csv", index=False)


if __name__ == '__main__':
    main()
