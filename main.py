from collections import defaultdict

import numpy as np
from surprise import SVD, KNNBasic
from surprise import Dataset
from surprise.model_selection import train_test_split
from sklearn.metrics import ndcg_score


def get_top_n(predictions, n):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


def get_testset_items_per_user(testset):
    items_per_user = defaultdict(dict)
    for uid, iid, true_r in testset:
        items_per_user[uid][iid] = true_r

    return items_per_user


def calc_ndcg(top_n, testset_items_per_user, n):
    y_trues = []
    y_scores = []

    for user in top_n:
        y_true = []
        y_score = []

        for item, est in top_n[user]:
            y_true.append(testset_items_per_user[user][item])
            y_score.append(est)

        # pad all arrays, if needed (for some users there arent as many item predictions)
        p = n - len(y_true)

        y_true = np.array(y_true)
        y_true = np.pad(y_true, [(0, p)], mode='constant', constant_values=0)
        y_trues.append(y_true)

        y_score = np.array(y_score)
        y_score = np.pad(y_score, [(0, p)], mode='constant', constant_values=0)
        y_scores.append(y_score)

    y_trues = np.array(y_trues)
    y_scores = np.array(y_scores)

    return ndcg_score(y_trues, y_scores, k=None)


if __name__ == '__main__':
    # movielens-100k dataset
    data = Dataset.load_builtin("ml-100k")

    # split data in train test
    trainset, testset = train_test_split(data, test_size=0.3)

    # reformat to user -> item -> score
    testset_items_per_user = get_testset_items_per_user(testset)

    # SVD
    algos = [
        # SVD
        (
            "SVD",
            SVD(n_factors=100,
                n_epochs=20,
                biased=True,
                init_mean=0,
                init_std_dev=0.1,
                lr_all=0.005,
                reg_all=0.02,
                verbose=True)
        ),

        # KNN
        (
            "KNN",
            KNNBasic(k=40,
                     min_k=1,
                     verbose=True)
        )
    ]

    # train algos
    for name, algo in algos:
        print(name)
        algo.fit(trainset)

    # test, limit to top N
    n = 10

    for name, algo in algos:
        # get item predictions
        predictions = algo.test(testset)

        # get top n predictions
        top_n = get_top_n(predictions, n=n)

        # calculate ndcg
        ndcg = calc_ndcg(top_n, testset_items_per_user, n=n)

        print("NDCG", name, ndcg)

    print("done")
