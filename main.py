import sklearn
import pandas as pd
import numpy as np
import cleansing
import loading
from data_preperation import prepare_data
from classification import find_best_models, use_estimators
from imputation import *
from standartisation import *
from wrappers import *
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from model_selection import target_features_split


def data_sets_preparation(load_from_csv=False):
    if load_from_csv:
        train = pd.read_csv('train_processed.csv')
        valid = pd.read_csv('valid_processed.csv')
        test = pd.read_csv('test_processed.csv')
    else:
        train = pd.read_csv('orig_train.csv')
        valid = pd.read_csv('orig_valid.csv')
        test = pd.read_csv('orig_test.csv')

        train = pd.concat([train, test], ignore_index=True)

        test = pd.read_csv("ElectionsData_Pred_Features.csv")
        id_num = test["IdentityCard_Num"]

        train, valid, test = prepare_data(train, valid, test)
        test["IdentityCard_Num"] = id_num
        test.to_csv('test_processed.csv', index=False)

    return train, valid, test


def main():
    train, valid, test = data_sets_preparation(False)

    features = train.columns.tolist()
    selected_features = ["Avg_environmental_importance", "Avg_government_satisfaction", "Avg_education_importance",
                         "Avg_monthly_expense_on_pets_or_plants", "Avg_Residancy_Altitude", "Yearly_ExpensesK",
                         "Weighted_education_rank", "Number_of_valued_Kneset_members"]
    features = [feat for feat in features if feat.startswith("Issue")] + selected_features

    best_models = find_best_models(train, valid, verbose=True, search_hyper_params=True)
    use_estimators(best_models, train, valid, test)

    results = pd.DataFrame()
    results["IdentityCard_Num"] = test["IdentityCard_Num"]
    results["PredictVote"] = pd.read_csv('test_predictions.csv', header=None)[0]
    results.to_csv("test_results.csv", index=False)

    test["Vote"] = results["PredictVote"]
    test[features + ['Vote']].to_csv("new_test_processed.csv", index=False)


if __name__ == '__main__':
    main()
