from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier

def train(X_train, y_train):
    xgb_model = MultiOutputClassifier(
        XGBClassifier(
            n_estimators=300,
            learning_rate=0.01,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=2
        )
    )
    xgb_model.fit(X_train, y_train)

    return xgb_model