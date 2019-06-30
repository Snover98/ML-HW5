import loading
import numpy as np
import pandas as pd
import sklearn
from imputation import *
import cleansing
from standartisation import *

""" the total preparing of the data:
    1. load the data and split it to train set,validation set, test_set
    2. cleanse the data from not logical examples
    3. impute missing values in the data
    4. leave only the selected features """


def prepare_data(train: pd.DataFrame = None, valid: pd.DataFrame = None, test: pd.DataFrame = None):
    if train is None or valid is None or test is None:
        df = loading.load_csv()

        # splitting the data
        train, valid, test = loading.split_data(df, 'Vote')

        # train.to_csv('orig_train.csv', index=False)
        # valid.to_csv('orig_valid.csv', index=False)
        # test.to_csv('orig_test.csv', index=False)

    test_has_labels = 'Vote' in test.columns

    selected_features = ["Avg_environmental_importance", "Avg_government_satisfaction", "Avg_education_importance",
                         "Avg_monthly_expense_on_pets_or_plants", "Avg_Residancy_Altitude", "Yearly_ExpensesK",
                         "Weighted_education_rank", "Number_of_valued_Kneset_members"]
    non_rand_features = selected_features + ["Married", "Most_Important_Issue", "Avg_Satisfaction_with_previous_vote",
                                             "Avg_monthly_household_cost", "Political_interest_Total_Score",
                                             "Phone_minutes_10_years", "Avg_size_per_room",
                                             "Avg_monthly_income_all_years", "Avg_monthly_expense_when_under_age_21",
                                             "AVG_lottary_expanses", "Last_school_grades"]

    train = train[['Vote'] + non_rand_features].copy()
    valid = valid[['Vote'] + non_rand_features].copy()
    test = test[(['Vote'] if test_has_labels else []) + non_rand_features].copy()

    # cleansing the data
    train = pd.DataFrame(cleansing.cleanse(train))
    valid = pd.DataFrame(cleansing.cleanse(valid))
    test = pd.DataFrame(cleansing.cleanse(test))

    # imputation of the data
    imputation(train)
    imputation(valid, train)
    if test_has_labels:
        imputation(test, train)
    else:
        new_imputation(test, train)

    features: List[str] = train.columns.to_numpy().tolist()
    features = [feat for feat in features if feat.startswith("Issue")] + selected_features

    scaler = DFScaler(train, selected_features)

    train = scaler.scale(train)
    valid = scaler.scale(valid)
    test = scaler.scale(test)

    train[features + ['Vote']].to_csv('train_processed.csv', index=False)
    valid[features + ['Vote']].to_csv('valid_processed.csv', index=False)
    test[features + (['Vote'] if test_has_labels else [])].to_csv('test_processed.csv', index=False)

    return train[features + ['Vote']], valid[features + ['Vote']], test[
        features + (['Vote'] if test_has_labels else [])]


if __name__ == "__main__":
    prepare_data()
