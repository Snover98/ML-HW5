import pandas as pd
import numpy as np

""" recieves the election data as input and replaces the values of categorial featuers with numeric representation
    and make a one hot column for some of the other features, and cahnge the type of every feature to float"""


def change_categorials(df: pd.DataFrame):
    cleanup_nums = {"Age_group": {"Below_30": 0, "30-45": 1, "45_and_up": 2},
                    "Looking_at_poles_results": {"Yes": 1, "No": 0},
                    "Married": {"Yes": 1, "No": 0},
                    "Financial_agenda_matters": {"Yes": 1, "No": 0},
                    "Gender": {"Male": 1, "Female": -1},
                    "Voting_Time": {"After_16:00": 1, "By_16:00": -1},
                    "Will_vote_only_large_party": {"Yes": 1, "Maybe": 0, "No": -1}
                    }
    df.replace(cleanup_nums, inplace=True)

    df = pd.get_dummies(df, columns=["Most_Important_Issue", "Main_transportation", "Occupation"],
                        prefix=["Issue", "trans", "Occ"])

    for col in df:
        if col == "Vote":
            df[col].astype('category', copy=False)
        else:
            df[col].astype(float, copy=False)

    return df


"""recieves the election data and a feature and a range of the feature and drops all exampels that
   with a feature that isnt in its range or isnt an int when it should be"""


def outlier_dropping(feature: int, df: pd.DataFrame, min: int = 0, max: int = np.inf, is_int: bool = False,
                     finite_range=None):
    to_drop = []
    for count, (idx, row) in enumerate(df.iterrows()):
        val = row[feature]
        if pd.isna(val):
            continue
        elif finite_range is not None and val not in finite_range:
            to_drop.append(idx)
        elif is_int and not np.isclose(val, int(val)):
            to_drop.append(idx)
        if val < min or val > max:
            to_drop.append(idx)

    df.drop(to_drop, inplace=True)
    return df


""" recieves the election data and preforms outlier_dropping for specific featuers that have constrains on them
    like range or being an int"""


def cleanse(df: pd.DataFrame):
    for count, feature in enumerate(df):
        if df[feature].dtype != float:
            continue
        elif feature in ["%Of_Household_Income", "%Time_invested_in_work", "%_satisfaction_financial_policy"]:
            outlier_dropping(count, df, 0, 100)
        elif feature == "Number_of_differnt_parties_voted_for":
            outlier_dropping(count, df, is_int=True)
        elif feature == "Financial_balance_score_(0-1)":
            outlier_dropping(count, df, 0, 1)
        elif feature == "Occupation_Satisfaction":
            outlier_dropping(count, df, finite_range=range(1, 11, 1))
        elif feature == "Number_of_valued_Kneset_members":
            outlier_dropping(count, df, finite_range=range(0, 121, 1))
        elif feature == "Num_of_kids_born_last_10_years":
            outlier_dropping(count, df, finite_range=range(0, 11, 1))
        elif feature == "Last_school_grades":
            outlier_dropping(count, df, finite_range=range(0, 101, 10))
        else:
            outlier_dropping(count, df)
    return change_categorials(df)
