import pandas as pd
from src.features.build_features import features_engineering

def test_features_engineering():
    data = {
        'Airline': ['IndiGo'],
        'Date_of_Journey': ['2025-01-01'],
        'Source': ['Banglore'],
        'Destination': ['New Delhi'],
        'Dep_Time': ['2025-01-01 10:00'],
        'Arrival_Time': ['2025-01-01 12:30'],
        'Duration': [150],
        'Total_Stops': [0],
        'Additional_Info': ['No info'],
        'Price': [3897],
    }
    df = pd.DataFrame(data)
    df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'])
    df['Dep_Time'] = pd.to_datetime(df['Dep_Time'])
    df['Arrival_Time'] = pd.to_datetime(df['Arrival_Time'])
    df = features_engineering(df)
    assert 'Date_of_Journey' not in df.columns
    assert 'Dep_Time' not in df.columns
    assert 'Arrival_Time' not in df.columns
    assert 'Additional_Info' not in df.columns
