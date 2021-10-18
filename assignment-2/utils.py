import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing


def load_grades_data():
    data = pd.read_csv('data/collegePlace.csv')
    dummy_gender = pd.get_dummies(data["Gender"])
    dummy_stream = pd.get_dummies(data["Stream"])
    data = pd.concat([data.drop(["Gender", "Stream"], axis=1),
                     dummy_gender, dummy_stream], axis=1)

    # reorder data
    data = data[['Age', 'Male', 'Female',
                 'Electronics And Communication',
                 'Computer Science', 'Information Technology',
                 'Mechanical', 'Electrical', "Civil",
                 "Internships", "CGPA", 'Hostel',
                 'HistoryOfBacklogs', 'PlacedOrNot']]

    X = data.drop('PlacedOrNot', axis=1)
    X = preprocessing.scale(X)
    X = pd.DataFrame(X, columns=data.columns[:-1])
    y = data['PlacedOrNot']
    return X, y


def load_heart_data():
    data = pd.read_csv('data/heart.csv')

    a = pd.get_dummies(data['cp'], prefix="cp")
    b = pd.get_dummies(data['thal'], prefix="thal")
    c = pd.get_dummies(data['slope'], prefix="slope")

    frames = [data, a, b, c]
    data = pd.concat(frames, axis=1)
    data = data.drop(columns=['cp', 'thal', 'slope'])
    X = data.drop('target', axis=1)
    X = preprocessing.scale(X)
    y = data['target']

    return X, y
