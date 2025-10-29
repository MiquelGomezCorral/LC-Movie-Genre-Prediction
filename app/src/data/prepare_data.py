import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import joblib

from src.config import Configuration

nltk.download('stopwords')
nltk.download('punkt_tab')
stemmer = PorterStemmer()
stop_words = stopwords.words('english')

def prepare_labels(df: pd.DataFrame, label: str) -> np.ndarray:
    df["labels"] = df[label].apply(lambda x: [l.strip() for l in x.split(",")])

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["labels"].tolist())
    joblib.dump(mlb,"models/binarizer.pkl")
    return y

def prepare_text_transformer(df: pd.DataFrame, cols: list):
    df["text"] = df[cols].apply(lambda x: " [SEP] ".join(x.dropna().astype(str)), axis=1)
    
    return df["text"].tolist()

def tokenizer_text(text):
    text = ''.join([re.sub(r'[^\w\s]', ' ', phrase) for phrase in text])
    text = word_tokenize(text)
    text = [stemmer.stem(word) for word in text if word.isalpha() and word not in stop_words]
    return text
    
def prepare_text(df: pd.DataFrame, cols: list, vectorizer = None) -> np.ndarray:
    df["text"] = df[cols].apply(lambda x: " ".join(x.dropna().astype(str)), axis=1)
    print(vectorizer)
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            analyzer='word',
            max_features=15000,
            ngram_range=(1, 2),
            lowercase=True,
            norm='l2',
            tokenizer=tokenizer_text,        
        )

        X = vectorizer.fit_transform(df["text"].tolist())
        joblib.dump(vectorizer, "models/tfidf.pkl")
        
    else:
        X = vectorizer.transform(df["text"].tolist())

    return X

def split(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    return X_train, X_val, y_train, y_val

def prepare_data_train(cfg: Configuration):
    df = pd.read_csv(cfg.train_data)

    X = prepare_text(df, cfg.columns)
    y = prepare_labels(df, cfg.label)

    X_train, X_val, y_train, y_val = split(X, y)

    return X_train, X_val, y_train, y_val

def prepare_data_train_transformer(cfg: Configuration):
    df = pd.read_csv(cfg.train_data)

    X = prepare_text_transformer(df, cfg.columns)
    y = prepare_labels(df, cfg.label)

    X_train, X_val, y_train, y_val = split(X, y)

    train_dataset = [
        {
            'text': text,
            'labels': label
        }
        for text, label in zip(X_train, y_train)
    ]
    val_dataset = [
        {
            'text': text,
            'labels': label
        }
        for text, label in zip(X_val, y_val)
    ]

    return train_dataset, val_dataset

def prepare_data_test(cfg: Configuration):
    df = pd.read_csv(cfg.test_data)
    
    vectorizer = joblib.load("models/tfidf.pkl")
    X_test = prepare_text(df, cfg.columns, vectorizer)
    return X_test

def prepare_data_test_transformer(cfg:Configuration):
    df = pd.read_csv(cfg.test_data)

    X_test = prepare_text_transformer(df, cfg.columns)
    return X_test
