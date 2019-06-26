import pandas as pd
from sklearn.model_selection import train_test_split


# load a dataset from a csv file to a panda dataframe
def load_csv():
    data = pd.read_csv("ElectionsData.csv")
    return data


# splits the given dataset to a train set,validation set and test set according to the given ratio
def split_data(data: pd.DataFrame, target: str, valid_ratio: float = 0.15, test_ratio: float = 0.1):
    train_ratio = 1 - (valid_ratio + test_ratio)
    assert train_ratio >= 0.5

    train, test = train_test_split(data, test_size=test_ratio, stratify=data[target])
    valid_ratio = valid_ratio / (1 - test_ratio)
    train, valid = train_test_split(train, test_size=valid_ratio, stratify=train[target])

    return train, valid, test
