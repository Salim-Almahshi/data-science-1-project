from collections import defaultdict
from copy import deepcopy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surprise import SVD, KNNBasic, KNNWithZScore, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from sklearn.metrics import ndcg_score

import streamlit as st

# to debug in PyCharm see:
# https://stackoverflow.com/questions/60172282/how-to-run-debug-a-streamlit-application-from-an-ide


def get_top_n(predictions, n, k):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        # filter by threshold
        if est >= k:
            top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        if n > 0:
            top_n[uid] = user_ratings[:n]
        else:
            top_n[uid] = user_ratings

    return top_n


def filter_predictions_to_top_n_and_threshold_k(predictions, n, k):
    filtered_predictions = []

    for pred in predictions:
        uid, iid, true_r, est, _ = pred
        if est >= k:
            filtered_predictions.append(pred)

            if n > 0 and len(filtered_predictions) >= n:
                break

    return filtered_predictions



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

def calc_map(predictions, k, threshold):

    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    map = 0.0
    user_number = 0

    for uid, user_ratings in user_est_true.items():
        user_number += 1
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        n_rel_and_rec_k = 0
        ap = 0.0 # for average precision
        for i in range(1,k+1):
            # Number of recommended items in top k
            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

            # Number of relevant and recommended items in top k
            n_rel_and_rec_k_new = sum(((true_r >= threshold) and (est >= threshold))
                                  for (est, true_r) in user_ratings[:k])

            if (n_rel_and_rec_k < n_rel_and_rec_k_new):
                n_rel_and_rec_k = n_rel_and_rec_k_new
                ap += n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        map += ap / k

    map = map / user_number
    return map


@st.cache
def load_data(name, random_state):
    # data from Surprise data loader
    data = []

    # pandas dataframe containing the data for visualizing
    # should contain rows of "user, item, rating, timestamp"
    dataframe = []

    # load MovieLens
    if name == "MovieLens":
        # load raw data via buildin surprise loader
        # data = Dataset.load_builtin("ml-1m")
        data = Dataset.load_builtin("ml-100k", prompt=False)

        # extract raw ratings to datframe for visualization
        # raw ratings are tuples of (user, item, rating, timestamp)
        dataframe = pd.DataFrame(data.raw_ratings, columns=["user", "item", "rating", "timestamp"])

        # convert timestamp to date
        dataframe['timestamp'] = pd.to_numeric(dataframe['timestamp'])
        dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'], unit='s')

    # load Restaurant Data
    elif name == "Restaurant":
        reader = Reader(rating_scale=(0,2))
        dataframe = pd.read_csv("rest_data.csv")
        dataframe.rename(columns={
            "userID": "user",
            "placeID": "item"
        }, inplace=True)
        data = Dataset.load_from_df(dataframe[["user", "item", "rating"]], reader = reader)

    # load Bookcrossing Data
    elif name == "BookCrossing":
        reader = Reader(rating_scale=(0, 10))
        dataframe = pd.read_excel('BX-Users-Rating-New.xlsx')
        dataframe.rename(columns={
            "User-ID": "user",
            "ISBN": "item",
            "Book-Rating": "rating"
        }, inplace=True)
        data = Dataset.load_from_df(dataframe[["user", "item", "rating"]], reader=reader)

    # split data in train test
    trainset, testset = train_test_split(data, test_size=0.3, random_state=random_state)

    # reformat to user -> item -> score
    testset_items_per_user = get_testset_items_per_user(testset)

    # to calculate threshold for map : little bigger than median of the rating range
    # "MovieLens", "BookCrossing", "Restaurant"
    # if option_dataset == "MovieLens":
    #    threshold = 3.5
    # elif option_dataset == "BookCrossing":
    #    threshold = 7.0
    # elif option_dataset == "Restaurant":
    #    threshold = 1.2
    map_threshold = np.mean(np.array(dataframe["rating"].to_list()))

    return {
        "trainset": trainset,
        "testset": testset,
        "testset_items_per_user": testset_items_per_user,
        "dataframe": dataframe,
        "map_threshold": map_threshold
    }


@st.cache(suppress_st_warning=True)
def train_model(settings, trainset):
    algos = {}

    for algo_name in settings:
        algo = {}

        if algo_name == "SVD":
            algo = SVD(n_factors=settings["SVD"]["n_factors"],
                       n_epochs=settings["SVD"]["n_epochs"],
                       biased=settings["SVD"]["biased"],
                       init_mean=settings["SVD"]["init_mean"],
                       init_std_dev=settings["SVD"]["init_std_dev"],
                       lr_all=settings["SVD"]["lr_all"],
                       reg_all=settings["SVD"]["reg_all"],
                       verbose=False)

        elif algo_name == "KNNBasic":
            algo = KNNBasic(k=settings["KNNBasic"]["k"],
                            min_k=settings["KNNBasic"]["min_k"],
                            sim_options={
                                "name": settings["KNNBasic"]["sim_name"],
                                "user_based": settings["KNNBasic"]["sim_user_based"],
                                "min_support": settings["KNNBasic"]["sim_min_support"],
                                "shrinkage": settings["KNNBasic"]["sim_shrinkage"]
                            },
                            verbose=False)

        elif algo_name == "KNNWithZScore":
            algo = KNNWithZScore(k=settings["KNNWithZScore"]["k"],
                                 min_k=settings["KNNWithZScore"]["min_k"],
                                 sim_options={
                                     "name": settings["KNNWithZScore"]["sim_name"],
                                     "user_based": settings["KNNWithZScore"]["sim_user_based"],
                                     "min_support": settings["KNNWithZScore"]["sim_min_support"],
                                     "shrinkage": settings["KNNWithZScore"]["sim_shrinkage"]
                                })

        algos[algo_name] = algo

        algo.fit(trainset)

    return algos


# set page title
st.title("Data Science 1 Project SS2021")

# add options to sidebar
st.sidebar.subheader("Options")

# dataset selection, can only select on
# option_dataset = st.sidebar.selectbox("Dataset", ["MovieLens", "BookCrossing"])
option_dataset = st.sidebar.selectbox("Dataset", ["MovieLens", "BookCrossing", "Restaurant"])
option_random_state = st.sidebar.number_input("Random state", min_value=0)

# evaluation settings
option_n = st.sidebar.number_input("Top N recommendations", min_value=1, value=5)
option_k = st.sidebar.number_input("Recommendation prediction threshold", min_value=0.0, max_value=1.0, value=0.5)

# algo selection, can select multiple
option_algos = st.sidebar.multiselect("Algorithms", ["SVD", "KNNBasic", "KNNWithZScore"])

# add algo settings to sidebar, depending on selected algo
options_algo_settings = {}
for algo in option_algos:
    st.sidebar.subheader(algo)

    if algo == "SVD":
        options_algo_settings[algo] = {
            "n_factors": st.sidebar.number_input("n factors", min_value=1, value=100, key=algo+"_n_factors_option_input"),
            "n_epochs": st.sidebar.number_input("n epochs", min_value=1, value=20, key=algo+"_min_n_epochs_option_input"),
            "biased": st.sidebar.checkbox("biased", value=True, key=algo+"_biased_option_input"),
            "init_mean": st.sidebar.number_input("init mean", value=0.0, key=algo+"_init_mean_epochs_option_input"),
            "init_std_dev": st.sidebar.number_input("init std dev", value=0.1, key=algo+"_init_std_dev_epochs_option_input"),
            "lr_all": st.sidebar.number_input("lr all", value=0.005, key=algo+"_lr_all_epochs_option_input"),
            "reg_all": st.sidebar.number_input("reg all", value=0.02, key=algo+"_reg_all_epochs_option_input"),
        }

    elif algo == "KNNBasic":
        options_algo_settings[algo] = {
            "k": st.sidebar.number_input("k", min_value=1, value=40, key=algo+"_k_option_input"),
            "min_k": st.sidebar.number_input("min k", min_value=1, value=1, key=algo+"_min_k_option_input"),
            "sim_name": st.sidebar.selectbox("similarity measure", ["msd", "cosine", "pearson", "pearson_baseline"], key=algo+"_sim_name_option_input"),
            "sim_user_based": st.sidebar.checkbox("user based", value=True, key=algo+"_sim_user_based_option_input"),
            # TODO default value?
            "sim_min_support": st.sidebar.number_input("min support", min_value=1, value=1, key=algo+"_sim_min_support_option_input"),
            "sim_shrinkage": st.sidebar.number_input("shrinkage", min_value=1, value=100, key=algo+"_sim_shrinkage_option_input"),
        }

    elif algo == "KNNWithZScore":
        options_algo_settings[algo] = {
            "k": st.sidebar.number_input("k", min_value=1, value=40, key=algo+"_k_option_input"),
            "min_k": st.sidebar.number_input("min k", min_value=1, value=1, key=algo+"_min_k_option_input"),
            "sim_name": st.sidebar.selectbox("similarity measure", ["msd", "cosine", "pearson", "pearson_baseline"], key=algo+"_sim_name_option_input"),
            "sim_user_based": st.sidebar.checkbox("user based", value=True, key=algo+"_sim_user_based_option_input"),
            # TODO default value?
            "sim_min_support": st.sidebar.number_input("min support", min_value=1, value=1, key=algo+"_sim_min_support_option_input"),
            "sim_shrinkage": st.sidebar.number_input("shrinkage", min_value=1, value=100, key=algo+"_sim_shrinkage_option_input"),
        }

st.subheader("Dataset:")
st.write(option_dataset)

data = deepcopy(load_data(option_dataset, option_random_state))

# show dataset stats
st.write("First lines of dataset:")
st.write(data["dataframe"].head(n=100))

st.write("Dataset statistics:")
st.write(data["dataframe"].describe(include='all'))

st.write("Dataset distribution of ratings:")
fig, ax = plt.subplots()
ratings_dist = data["dataframe"]["rating"].value_counts()
ax.bar(ratings_dist.index.to_list(), ratings_dist)
ax.set_xlabel("Rating")
ax.set_ylabel("Count")
st.pyplot(fig)

st.write("Dataset distribution of ratings per user:")
fig, ax = plt.subplots()
user_ratings_counts = data['dataframe'].groupby(by=['user'])["rating"].count()
ax.hist(user_ratings_counts)
ax.set_xlabel("Rating count")
ax.set_ylabel("User count")
st.pyplot(fig)

st.subheader("Algorithms:")
st.write(", ".join(option_algos))

algos = train_model(options_algo_settings, data["trainset"])

# evaluate algos
metrics = {}
for algo in algos:
    st.write("Evaluating", algo)

    # get item predictions
    predictions = algos[algo].test(data["testset"])
    filtered_predictions = filter_predictions_to_top_n_and_threshold_k(predictions, option_n, option_k)

    # get top n predictions
    top_n = get_top_n(predictions, n=option_n, k=option_k)

    # calculate metrics
    metrics[algo] = {
        "ndcg": calc_ndcg(top_n, data["testset_items_per_user"], n=option_n),
        "mse": accuracy.mse(filtered_predictions),
        "rmse": accuracy.rmse(filtered_predictions),
        "mae": accuracy.mae(filtered_predictions),
        #"fcp": accuracy.fcp(filtered_predictions),
        "map": calc_map(predictions, option_n, data["map_threshold"])
    }

all_metrics = pd.DataFrame(metrics)
st.write(all_metrics)