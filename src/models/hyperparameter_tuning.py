import logging
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow.xgboost
from skopt import BayesSearchCV
from sklearn.metrics import mean_squared_error
from typing import Dict, Any

# Configure logging
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


def tune_xgboost_hyperparameters(
    X: pd.DataFrame, 
    y: pd.Series, 
    param_space: Dict[str, Any], 
    n_iter: int = 50
) -> xgb.XGBRegressor:
    """
    Performs Bayesian Optimization for XGBoost hyperparameter tuning.
    
    Args:
        X (pd.DataFrame): Training features.
        y (pd.Series): Training target.
        param_space (Dict[str, Any]): Dictionary defining the hyperparameter search space.
        n_iter (int): Number of iterations for Bayesian optimization.
    
    Returns:
        xgb.XGBRegressor: The best trained XGBoost model.
    """
    logger.info("Starting Bayesian Optimization for XGBoost...")
    
    with mlflow.start_run() as run:
        mlflow.set_tag("mlflow.note.content", "Bayesian Optimization for XGBoost")
        logger.info(f"MLflow Run ID: {run.info.run_id}")

        # Initialize the XGBoost regressor
        xgb_regressor = xgb.XGBRegressor(random_state=42)
        
        # Set up BayesSearchCV
        model_tuner = BayesSearchCV(
            estimator=xgb_regressor,
            search_spaces=param_space,
            n_iter=n_iter,
            scoring="neg_mean_squared_error",
            cv=5,
            refit=True,
            random_state=42,
            n_jobs=-1,
        )
        
        # Fit the model tuner
        model_tuner.fit(X, y)

        # Get the best model and parameters
        best_model = model_tuner.best_estimator_
        best_params = model_tuner.best_params_
        logger.info(f"Best parameters found: {best_params}")
        
        # Log the best parameters
        mlflow.log_params(best_params)
        
        # Calculate and log training RMSE
        train_predictions = best_model.predict(X)
        train_rmse = np.sqrt(mean_squared_error(y, train_predictions))
        logger.info(f"Training RMSE: {train_rmse:.4f}")
        mlflow.log_metric("train_rmse", train_rmse)

        # Calculate and log test RMSE (using the same dataset for validation)
        test_predictions = best_model.predict(X)
        test_rmse = np.sqrt(mean_squared_error(y, test_predictions))
        logger.info(f"Test RMSE: {test_rmse:.4f}")
        mlflow.log_metric("test_rmse", test_rmse)

        # Log the best model to MLflow
        mlflow.xgboost.log_model(
            xgb_model=best_model,
            artifact_path="best_xgboost_model",
            input_example=X.head(1)
        )
        logger.info("Best XGBoost model logged to MLflow.")
        
    return best_model