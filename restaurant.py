# restaurant.py
# to drop some columns in original data "rating_final.csv"
# to get the name of restaurants

import pandas as pd

rating = pd.read_csv("rating_final.csv")
restaurant = pd.read_csv("geoplaces2.csv")
restaurant = restaurant[["placeID", "name"]]
rating.drop(['food_rating', 'service_rating'], axis=1, inplace= True)

rating_restaurant = pd.merge(rating, restaurant, on="placeID")

rating_restaurant.to_csv("rest_data.csv", index = False)

