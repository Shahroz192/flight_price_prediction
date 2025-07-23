import pandas as pd
import numpy as np
from src.models.predict_model import load_data, predict, evaluate
from sklearn.linear_model import LinearRegression


def test_load_data():
    data = {"Price": [1, 2, 3], "A": [4, 5, 6]}
    df = pd.DataFrame(data)
    df.to_csv("test_data.csv", index=False)
    X, y = load_data("test_data.csv")
    assert len(X) == 3
    assert len(y) == 3


def test_predict():
    data = {"A": [1, 2, 3]}
    X = pd.DataFrame(data)
    model = LinearRegression()
    model.fit(X, [1, 2, 3])
    predictions = predict(X, model)
    assert len(predictions) == 3


def test_evaluate():
    y = pd.Series([1, 2, 3])
    predictions = np.array([1, 2, 3])
    mse, mae, r2, mape, med = evaluate(y, predictions)
    assert mse == 0
    assert mae == 0
    assert r2 == 1
    assert mape == 0
    assert med == 0
