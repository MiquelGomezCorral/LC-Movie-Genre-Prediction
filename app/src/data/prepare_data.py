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

def prepare_labels(labels: list) -> np.ndarray:
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(labels)

    return y

nltk.download('stopwords')
stemmer = PorterStemmer()
stop_words = stopwords.words('english')

def tokenizer_text(text):
    text = ''.join([re.sub(r'[^\w\s]', ' ', phrase) for phrase in text])
    text = word_tokenize(text)
    text = [stemmer.stem(word) for word in text if word.isalpha() and word not in stop_words]
    return text
    
def prepare_text(text: list) -> np.ndarray:
    vectorizer = TfidfVectorizer(
        analyzer='word',
        max_features=15000,
        ngram_range=(1, 2),
        lowercase=True,
        norm='l2',
        tokenizer=tokenizer_text,        
    )

    X = vectorizer.fit_transform(text)

    return X

def prepare_data_train(path: str, bag_X, label: str):
    df = pd.read_csv(path)

    df["text"] = df[bag_X].apply(lambda x: " ".join(x.dropna().astype(str)), axis=1)
    df["labels"] = df[label].apply(lambda x: [l.strip() for l in x.split(",")])

    print(df[["text","labels"]])

    X = prepare_text(df["text"].tolist())
    y = prepare_labels(df["labels"].tolist())

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    return X_train, X_val, y_train, y_val

    