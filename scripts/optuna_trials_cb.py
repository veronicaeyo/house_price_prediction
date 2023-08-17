from first_preprocess import first_preprocess

import optuna
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import os

from dotenv import load_dotenv

# Load environment variables from a file named ".env"
load_dotenv()


X, y, X_test = first_preprocess(
    "../data/Housing_dataset_train.csv",
    "../data/Housing_dataset_test.csv",
    "../json/population_density.json",
)


X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=7)


def objective(trial):
    params = {
        "loss_function": "RMSE",
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 2, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "thread_count": os.cpu_count(),
        "verbose": False,
    }

    model = CatBoostRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,
        verbose=False,
    )

    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)

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
        study_name="catboost_house_price",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
