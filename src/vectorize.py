import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
import os

output_dir = "../artifacts"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df = pd.read_csv("../output/processed_reviews.csv")

x_train, x_test, y_train, y_test = train_test_split(df['text'], df['rating'], test_size=0.2, random_state=42, stratify=df['rating'])

tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.8, ngram_range=(1, 2))

x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)

x_test_tfidf = tfidf_vectorizer.transform(x_test)

joblib.dump(x_train_tfidf, "../artifacts/x_train_tfidf.pkl")
joblib.dump(x_test_tfidf, "../artifacts/x_test_tfidf.pkl")

joblib.dump(y_train, "../artifacts/y_train.pkl")
joblib.dump(y_test, "../artifacts/y_test.pkl")

joblib.dump(tfidf_vectorizer, "../artifacts/tfidf_vectorizer.pkl")