import pandas as pd
from src.data.make_dataset import (
    remove_outliers,
    replacement,
    stop,
    arrival_time,
    duration,
    to_datetime,
    cleaning,
)

def test_remove_outliers():
    data = {'A': [1, 2, 3, 4, 5, 100], 'B': [1, 2, 3, 4, 5, 6]}
    df = pd.DataFrame(data)
    df = remove_outliers(df, ['A'])
    assert len(df) == 5

def test_replacement():
    data = {'Airline': ['Jet Airways Business'], 'Additional_Info': ['No Info']}
    df = pd.DataFrame(data)
    df = replacement(df)
    assert df['Airline'][0] == 'Jet Airways '
    assert df['Additional_Info'][0] == 'No info'

def test_stop():
    data = {'Total_Stops': ['non-stop', '1 stop', '2 stops']}
    df = pd.DataFrame(data)
    df = stop(df)
    assert df['Total_Stops'][0] == 0
    assert df['Total_Stops'][1] == 1
    assert df['Total_Stops'][2] == 2

def test_arrival_time():
    data = {'Arrival_Time': ['10:00 22 Mar']}
    df = pd.DataFrame(data)
    df = arrival_time(df)
    assert df['Arrival_Time'][0] == '10:00'

def test_duration():
    data = {'Duration': ['2h 30m', '2h', '30m']}
    df = pd.DataFrame(data)
    df = duration(df)
    assert df['Duration'][0] == 150
    assert df['Duration'][1] == 120
    assert df['Duration'][2] == 30

def test_to_datetime():
    data = {
        'Date_of_Journey': ['01/01/2025'],
        'Dep_Time': ['10:00'],
        'Arrival_Time': ['12:30'],
    }
    df = pd.DataFrame(data)
    df = to_datetime(df)
    assert pd.api.types.is_datetime64_any_dtype(df['Date_of_Journey'])

def test_cleaning():
    data = {
        'Airline': ['IndiGo'],
        'Date_of_Journey': ['01/01/2025'],
        'Source': ['Banglore'],
        'Destination': ['New Delhi'],
        'Route': ['BLR → DEL'],
        'Dep_Time': ['10:00'],
        'Arrival_Time': ['12:30'],
        'Duration': ['2h 30m'],
        'Total_Stops': ['non-stop'],
        'Additional_Info': ['No info'],
        'Price': [3897],
    }
    df = pd.DataFrame(data)
    df = cleaning(df)
    assert 'Route' not in df.columns
