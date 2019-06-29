import generative
import sklearn
import pandas as pd
import numpy as np
import clustering
from sklearn.metrics import davies_bouldin_score, completeness_score


def show_col_parties(coalition: pd.Series, labels: pd.Series):
    col_number = coalition.value_counts().idxmax()
    print('The coalition voters are:')
    print(labels[coalition == col_number].value_counts().sort_index())
    print('Out of:')
    print(labels.value_counts().sort_index())
    print(f'The coalition parties are {labels[coalition == col_number].unique()}')
    print(f'The Coalition has {100 * len(labels[coalition == col_number]) / len(coalition)}% of the votes')


def main():
    data = pd.read_csv("new_test_processed.csv")
    X, y = generative.target_features_split(data, "Vote")

    # cluster models
    print('Doing clustering coalitions')
    cluster_models = clustering.get_clustering(X, y)
    cluster_coalitions = clustering.create_cluster_coalitions(cluster_models, X, y, threshold=0.3)

    # generative_models
    print('Doing generative coalitions')
    gen_models = generative.train_generative(data)
    gen_coalitions = generative.create_gen_coalitions(gen_models, X, y)

    coalitions = cluster_coalitions + gen_coalitions

    model_names = ['MiniBatchKMeans', 'BayesianGMM', 'LDA', 'QDA']
    for model, name in zip(cluster_models + gen_models, model_names):
        clustering.show_clusters(X, model, f'{name} Clusters In 3D PCA Values')

    # check how good the coalitions are
    scores = []
    for coalition, name in zip(coalitions, model_names):
        col = y.isin(coalition).astype(np.int)

        scores.append(davies_bouldin_score(X, col))
        print('')
        print('=========================================')
        print(f'{name} Coalition')
        print(f'Score is {scores[-1]}')
        print(f'Completeness is {completeness_score(y, col)}')
        show_col_parties(pd.Series(col), y)
        clustering.show_labels(X, pd.Series(col), f'{name} Coalition')

    best_idx = np.argmin(scores)
    print(f'The best model is {model_names[best_idx]}')
    # pd.Series(coalitions[best_idx]).to_csv('coalition.csv', index=False)


if __name__ == '__main__':
    main()
