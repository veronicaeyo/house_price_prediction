import json

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess(train_data_path: str, test_data_path: str, json_path: str):
    df_train = pd.read_csv(train_data_path).drop("ID", axis=1)
    df_test = pd.read_csv(test_data_path).drop("ID", axis=1)

    # concatenate
    df_test["price"] = "test"
    df = pd.concat([df_train, df_test], axis=0)

    # add region column
    with open(json_path, "r") as f:
        df["region"] = df["loc"].map(json.load(f))

    # bedroom to bathroom ratio
    df["bed_bath_ratio"] = df["bedroom"] / df["bathroom"]

    # encode title column by ranks based on highest mean prices 
    title_ranks = (
        df_train.groupby("title")["price"]
        .mean()
        .sort_values(ascending=False)
        .rank(method="dense")
        .astype(int)
    )
    df['title_rank'] = df['title'].map(title_ranks)
    df.drop('title', axis=1, inplace=True)

    # rearrange to ensure 'price' is the last column
    price_col = df.pop('price')
    df.insert(len(df.columns), 'price', price_col)
    
    # split
    df_train = df[~df["price"].astype(str).str.contains("test")]
    df_test = df[df["price"].astype(str).str.contains("test")]

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    # fill null values
    imp_mode = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    imp_mode.fit(df_train)

    df_train = pd.DataFrame(imp_mode.transform(df_train), columns=df.columns)
    df_test = pd.DataFrame(imp_mode.transform(df_test), columns=df.columns)
    
    # label encoding
    le = LabelEncoder()

    for col in ["loc", "region"]:
        le = le.fit(df_train[col])
        df_train[col] = le.transform(df_train[col])
        df_test[col] = le.transform(df_test[col])
    
    # another split
    X = df_train.drop("price", axis=1)
    y = df_train["price"].astype(float)
    X_test = df_test.drop("price", axis=1)
    y_test = df_test["price"]
    
    # scale the dataset with standard scaler
    scaler = StandardScaler()
    train_scaler = scaler.fit(X)
    X = train_scaler.transform(X)
    X_test = train_scaler.transform(X_test)

    # make X a dataframe again 
    X = pd.DataFrame(X, columns=df_train.columns[0:-1])

    # split into train and validation set
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=7)
    
    return X_train, X_val, y_train, y_val
