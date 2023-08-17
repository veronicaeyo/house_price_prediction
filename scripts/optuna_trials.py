from preprocess import preprocess

import optuna
import lightgbm as lgbm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import os

from dotenv import load_dotenv

# Load environment variables from a file named ".env"
load_dotenv()


X, y, X_test = preprocess(
    "../data/Housing_dataset_train.csv",
    "../data/Housing_dataset_test.csv",
    "../json/state_to_region.json",
)


X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=7)


def objective(trial):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "n_estimators": 1000,
        "verbosity": -1,
        "bagging_freq": 1,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 2**10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
    }

    model = lgbm.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgbm.callback.early_stopping(stopping_rounds=100)],
    )

    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds, squared=False)

    return rmse


DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

if __name__ == "__main__":
    DB_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@localhost:{DB_PORT}/{DB_NAME}"
    study = optuna.create_study(
        direction="minimize",
        storage=DB_URI,
        study_name="LightGBM_house_price",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100)
