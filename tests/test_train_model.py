import pandas as pd
import xgboost as xgb
from src.models.train_model import train_and_tune, save_model_locally
from skopt.space import Real, Integer
import tempfile
import os

def test_train_and_tune():
    data = {'Price': [1, 2, 3, 4, 5, 6], 'A': [4, 5, 6, 7, 8, 9]}
    df = pd.DataFrame(data)
    X = df.drop('Price', axis=1)
    y = df['Price']
    param_space = {
        "n_estimators": Integer(100, 1000),
        "max_depth": Integer(3, 10),
        "learning_rate": Real(0.01, 0.3, prior='log-uniform'),
    }
    model = train_and_tune(X, y, param_space)
    assert isinstance(model, xgb.XGBRegressor)

def test_save_model_locally():
    model = xgb.XGBRegressor()
    with tempfile.NamedTemporaryFile(suffix=".joblib") as tmpfile:
        save_model_locally(model, tmpfile.name)
        assert os.path.exists(tmpfile.name)
