from sklearn.metrics import accuracy_score, hamming_loss, precision_recall_fscore_support

def compute_metrics(labels, preds):
    precision, recall, f1, _ = precision_recall_fscore_support(
    labels, preds, average="macro", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    hl = hamming_loss(labels, preds)
    return {
    "accuracy": acc,
    "f1": f1,
    "precision": precision,
    "recall": recall,
    "hamming_loss": hl
    }

def predict_and_metrics(model, X_val, y_val):
    y_pred = model.predict(X_val)

    metrics = compute_metrics(y_val, y_pred)

    return metrics