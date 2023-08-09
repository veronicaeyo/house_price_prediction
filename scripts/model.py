from typing import List, Callable

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def fit_score(model: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> BaseEstimator:
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=7)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    rmse = round(mean_squared_error(y_val, y_pred, squared=False))
    print(f"RMSE: {rmse:,}")

    model.fit(X, y)

    return model


def fold_fit_score(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    n_folds: int,
    transform_func: Callable,
    inverse_transform_func: Callable,
    plot_feat_imp=False,
) -> List[list]:
    rmse_list_transformed = []
    test_preds_transformed = []

    # create a StratifiedKFold object
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=20)

    # iterate over the folds
    for i, (train_index, val_index) in enumerate(kf.split(X, y)):
        # split the data into training and valing sets
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = transform_func(y[train_index].astype(float)), y[val_index]

        # fit the model on the training data
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        # evaluate the model on the validation data
        rmse = mean_squared_error(y_val, inverse_transform_func(y_pred), squared=False)
        rmse_list_transformed.append(rmse)

        # print the score for each fold
        print(f"Fold {i+1} RMSE: {rmse:,.0f}")
        print("=======" * 10)

        preds = model.predict(X_test)
        test_preds_transformed.append(inverse_transform_func(preds))

    print(f"Average RMSE: {np.mean(rmse_list_transformed):,.0f}")

    if plot_feat_imp:
        # feature importance plot for model
        feat_imp = pd.DataFrame(
            {"Feature": X.columns, "Importance": model.feature_importances_}
        )
        feat_imp = feat_imp.sort_values(by="Importance", ascending=False).reset_index(
            drop=True
        )
        plt.figure(figsize=(10, 8))
        sns.barplot(x="Importance", y="Feature", data=feat_imp)
        plt.title(f"Feature Importance for {type(model).__name__}")
        plt.show()

    return test_preds_transformed

if __name__ == "__main__":
    fit_score()
    fold_fit_score()