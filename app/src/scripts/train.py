from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
import joblib

def train(X_train, y_train):
    xgb_model = OneVsRestClassifier(
        LogisticRegression(
            max_iter=50000,
            C=0.7, 
            class_weight='balanced',
            solver='liblinear',
            tol=1e-5
        ), 
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    joblib.dump(xgb_model, "models/logistic_regression.pkl")
    return xgb_model
