import joblib
import pandas as pd

def test(X_test):
    model = joblib.load("models/logistic_regression.pkl")
    mlb = joblib.load("models/binarizer.pkl")

    y_pred = model.predict(X_test)
    print(y_pred)
    y_pred = mlb.inverse_transform(y_pred)
    y_pred = [", ".join(pred) if pred else "" for pred in y_pred]
    return y_pred
    
