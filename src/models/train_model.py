import logging
import pandas as pd
import xgboost as xgb
import joblib
import warnings
import mlflow
import mlflow.xgboost
from skopt import BayesSearchCV
from skopt.space import Real, Integer

warnings.filterwarnings("ignore", category=DeprecationWarning)


def objective(params, X, y):
    model = xgb.XGBRegressor(**params)
    return -1 * model.score(X, y)


def train(df):
    X = df.drop(["Price"], axis=1)
    y = df["Price"]

    with mlflow.start_run():
        # Define the hyperparameter search space
        param_space = {
            "base_score": Real(0.0, 1.0),
            "colsample_bylevel": Real(0.1, 1.0),
            "colsample_bynode": Real(0.1, 1.0),
            "colsample_bytree": Real(0.1, 1.0),
            "gamma": Real(0.0, 1.0),
            "learning_rate": Real(0.01, 0.3),
            "max_depth": Integer(2, 10),
            "min_child_weight": Integer(1, 10),
            "reg_alpha": Real(0.0, 1.0),
            "reg_lambda": Real(0.0, 1.0),
            "subsample": Real(0.1, 1.0),
        }

        # Perform Bayesian optimization
        model = BayesSearchCV(
            xgb.XGBRegressor(),
            param_space,
            n_iter=50,
            scoring="neg_mean_squared_error",
            cv=5,
            refit=True,
            random_state=42,
        )
        model.fit(X, y)

        # Log model parameters
        mlflow.log_params(model.best_params_)

        # Log model performance metrics
        train_rmse = model.score(X, y) * -1  # Negate the score as it was minimized
        mlflow.log_metric("train_rmse", train_rmse)

        # Log the model
        mlflow.xgboost.log_model(model.best_estimator_, "model")

    return model.best_estimator_


def save(model, path):
    try:
        joblib.dump(model, path)
        logging.info(f"Model saved to {path}")

        # Log the model artifact
        mlflow.log_artifact(path)

    except Exception as e:
        logging.error(f"Error saving the model: {e}")
        return None


def main():
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    mlflow.set_experiment("Flight Price Prediction")  # Set your experiment name

    try:
        df = pd.read_csv("data/processed/train_features.csv")
    except Exception as e:
        logging.error(f"Error loading the data: {e}")
        return
    logging.info("model training started")
    model = train(df)
    logging.info("Model trained successfully")
    save(model, "models/best_model")


if __name__ == "__main__":
    main()
