import pandas as pd
import numpy as np


# preforms a simple imputation by median
def median_imputation(feature: str, df: pd.DataFrame, df2: pd.DataFrame = None):
    # complete the missing values of a feature based on the avg of what all the other samples that voted the same
    if df[feature].dtype == float:
        for voting in df["Vote"].unique():
            if df2 is None:
                ss = df.groupby("Vote")[feature].mean()[voting]
            else:
                ss = df2.groupby("Vote")[feature].mean()[voting]
            df.loc[(df["Vote"] == voting) & (df[feature].isnull()), feature] = ss
    else:
        # complete the missing categorical value with the category most samples have
        if df2 is None:
            common_val = df[feature].value_counts().idxmax()
        else:
            common_val = df2[feature].value_counts().idxmax()
        df[feature].fillna(common_val, inplace=True)


# given the index of a sample in a dataset, a feature and the dataset(df) return the index of the sample closest to
# the given sample in the dataset on the given feature, if the second dataset is given too the closest sample
# will be searched on that dataset
def close_nieg(sample_idx: int, feature: int, df: pd.DataFrame, df2: pd.DataFrame = None):
    sample_val = df.iloc[sample_idx, feature]
    min_dist = np.inf
    min_nigh = -1

    if df2 is None:
        col = df.iloc[:, feature]
    else:
        col = df2.iloc[:, feature]
    col = pd.DataFrame(col)

    if df2 is None:
        for count, (idx, row) in enumerate(col.iterrows()):
            if count != sample_idx and abs(row[0] - sample_val) < min_dist:
                min_dist = abs(row[0] - sample_val)
                min_nigh = count

    else:
        for count, (idx, row) in enumerate(col.iterrows()):
            if abs(row[0] - sample_val) < min_dist:
                min_dist = abs(row[0] - sample_val)
                min_nigh = count

    return min_nigh


# given a feature and a dataset imputates a missing value of the feature  on each sample on the dataset
# with the value of the closest sample on the feature most correlated to the given feature
# on its value for the given feature, if there are no featuers strongly correlated to the given feature do nothing,
# if example has no value on the mosr correlated feature we will check the next one and if there are no more we will
# skip the example, given a second dataset we will imputate the data based on the second dataset examples
def related_features_imputation(feature: int, df: pd.DataFrame, df2: pd.DataFrame = None):
    df_tag = (df if df2 is None else df2).dropna()
    df_tag = df_tag.drop(["Vote"], axis=1)
    mi_matrix = df_tag.corr().as_matrix()

    feature -= 1

    feat_names = list(df)

    max_corr = [(i, mi_matrix[i][feature]) for i in range(mi_matrix.shape[0]) if
                i != feature and df.dtypes[feat_names[i + 1]] == float and abs(mi_matrix[i][feature]) > 0.5]

    max_corr.sort(reverse=True, key=lambda tup: tup[1])

    # if no correlation with any feature do nothing
    if len(max_corr) == 0:
        return -1

    for count, (idx, row) in enumerate(df.iterrows()):
        if pd.isna(row[feature]):
            for most_corr_feat, best_corr in max_corr:
                # check the sample that is closest in the correlated feature to the idx sample
                if pd.isna(row[most_corr_feat]):
                    continue
                if df2 is None:
                    close_nieg_index = close_nieg(count, most_corr_feat, df)
                    df.iloc[count, feature] = df.iloc[close_nieg_index, feature]
                else:
                    close_nieg_index = close_nieg(count, most_corr_feat, df, df2)
                    df.iloc[count, feature] = df2.iloc[close_nieg_index, feature]
                break


# imputate the dataset first according to correlated featuers and then complete the rest with median imputation
# if a second dataset is given too imputate the data on the first dataset according to the second dataset values
def imputation(dataset, dataset2=None):
    has_na = dataset.isna().any()
    # do a median_imputation
    for idx, col in enumerate(dataset):
        if has_na[col]:
            print(idx, col)
            related_features_imputation(idx, dataset, dataset2)
            median_imputation(col, dataset, dataset2)
