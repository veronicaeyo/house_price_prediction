import os
import warnings
import json
from typing import Union

import numpy as np
import pandas as pd
import category_encoders as ce

# Ignore all warnings
warnings.filterwarnings("ignore")


def first_preprocess(
    train_data_path: Union[str, os.PathLike],
    test_data_path: Union[str, os.PathLike],
    json_path: Union[str, os.PathLike],
):
    df_train = pd.read_csv(train_data_path)
    df_test = pd.read_csv(test_data_path)

    # create a new index column in df_train and df_test
    df_train["ind"] = 1
    df_test["ind"] = 0

    # Concatenate train and test DataFrames
    combined_df = pd.concat([df_train, df_test])

    # drop location and title column from combined_df
    combined_df.dropna(subset=["loc", "title"], inplace=True)

    # fill null values in combined_df
    median_bedroom = combined_df["bedroom"].median()
    combined_df["bedroom"].fillna(median_bedroom, inplace=True)

    median_bathroom = combined_df["bathroom"].median()
    combined_df["bathroom"].fillna(median_bathroom, inplace=True)

    median_parking_space = combined_df["parking_space"].median()
    combined_df["parking_space"].fillna(median_parking_space, inplace=True)

    # feature engineering

    combined_df["is_lagos"] = combined_df["loc"].apply(
        lambda x: 1 if x == "Lagos" else 0
    )

    combined_df["is_mansion"] = combined_df["title"].apply(
        lambda x: 1 if x == "Mansion" else 0
    )

    combined_df["comfort_ind"] = combined_df["bedroom"] / combined_df["bathroom"]

    combined_df["size"] = (
        combined_df["bedroom"] + combined_df["bathroom"] + combined_df["parking_space"]
    )

    combined_df["comfort_by_size"] = combined_df["comfort_ind"] * combined_df["size"]

    with open(json_path, "r") as f:
        population_levels = json.load(f)

    combined_df["population_density_level"] = combined_df["loc"].apply(
        lambda x: next((int(k) for k, v in population_levels.items() if x in v), 0)
    )

    # Create a new DataFrame to store the encoded values
    encoded_df = combined_df.copy()

    # Define the encoder instance
    encoder = ce.TargetEncoder(cols=["loc"])

    # Fit the encoder on the 'loc' column and 'price' target variable
    encoder.fit(encoded_df["loc"], encoded_df["price"])

    # Transform the 'loc' column with the encoded values
    encoded_df["loc_encoded"] = encoder.transform(encoded_df["loc"])

    encoded_df["loc_encoded"] = round((encoded_df["loc_encoded"] / 10_000_000), 2)
    # Drop the original 'loc' column if you no longer need it
    encoded_df.drop("loc", axis=1, inplace=True)

    # Calculate the average price for each title
    title_average_price = combined_df.groupby("title")["price"].mean()

    # Create a new DataFrame to store the title and its corresponding average price
    title_avg_price_df = pd.DataFrame(
        {"title": title_average_price.index, "avg_price": title_average_price.values}
    )

    # Sort the DataFrame by the average price in ascending order
    title_avg_price_df.sort_values(by="avg_price", inplace=True)

    # Create a new column 'title_rank' with the rank based on average price
    title_avg_price_df["title_rank"] = range(1, len(title_avg_price_df) + 1)

    # Merge the 'title_rank' column back to the original DataFrame based on 'title' column
    encoded_df = pd.merge(
        encoded_df, title_avg_price_df[["title", "title_rank"]], on="title", how="left"
    )

    # Drop the 'title_rank' column if you no longer need it
    encoded_df.drop("title", axis=1, inplace=True)

    combined_df = encoded_df.copy()

    train = combined_df[combined_df.ind == 1]
    test = combined_df[combined_df.ind == 0]

    train.drop("ind", axis=1, inplace=True)
    test.drop(["ind", "price"], axis=1, inplace=True)

    X = train.drop(["ID", "price"], axis=1)
    y = train.price
    
    X_test = test.drop(['ID'], axis=1)

    return X, y, X_test


if __name__ == "__main__":
    first_preprocess()
