import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "output", "processed_reviews.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "..", "artifacts")

def setup_environment(path):
    os.makedirs(path, exist_ok=True)

#загрузка и разбиение данных
def prepare_datasets(path):
    df = pd.read_csv(path)
    df['text'] = df['text'].fillna('')
    x_train, x_test, y_train, y_test = train_test_split(
        df['text'], 
        df['rating'], 
        test_size=0.2, 
        random_state=42, 
        stratify=df['rating']
    )
    return x_train, x_test, y_train, y_test

def vectorize_text(x_train, x_test, save_path):
    vectorizer = TfidfVectorizer(
        max_features=5000, 
        min_df=5, 
        max_df=0.8, 
        ngram_range=(1, 2)
    )
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)
    joblib.dump(vectorizer, os.path.join(save_path, "tfidf_vectorizer.pkl"))
    return x_train_tfidf, x_test_tfidf

def save_processed_data(x_tr, x_ts, y_tr, y_ts, save_path):
    joblib.dump(x_tr, os.path.join(save_path, "x_train_tfidf.pkl"))
    joblib.dump(x_ts, os.path.join(save_path, "x_test_tfidf.pkl"))
    joblib.dump(y_tr, os.path.join(save_path, "y_train.pkl"))
    joblib.dump(y_ts, os.path.join(save_path, "y_test.pkl"))

if __name__ == "__main__":
    setup_environment(ARTIFACTS_DIR)
    xtr_raw, xts_raw, ytr, yts = prepare_datasets(DATA_PATH)
    xtr_tfidf, xts_tfidf = vectorize_text(xtr_raw, xts_raw, ARTIFACTS_DIR)
    save_processed_data(xtr_tfidf, xts_tfidf, ytr, yts, ARTIFACTS_DIR)