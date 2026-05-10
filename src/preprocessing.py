import pandas as pd
import spacy
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import glob

#перед запуском еще нужно скачать библиотеку слов python -m spacy download en_core_web_sm но вроде она через requirements установится

DATA_DIR = "../data/"

files = glob.glob(os.path.join(DATA_DIR, "*.jsonl"))
INPUT_PATH = files[0]
OUTPUT_DIR = "../output"
STATS_PATH = os.path.join(OUTPUT_DIR, "stats.txt")
CSV_PATH = os.path.join(OUTPUT_DIR, "processed_reviews.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

def save_to_log(message):
    with open(STATS_PATH, "a", encoding="utf-8") as f:
        f.write(str(message) + "\n")

#згрузка данных и первичная чистка дублей
def load_data(path):
    df = pd.read_json(path, lines=True)
    save_to_log(f"Оригинальное количество строк: {len(df)}")
    
    df = df[["rating", "text"]].copy()
    df = df.dropna()
    df.drop_duplicates(keep='first', inplace=True)
    save_to_log(f"После удаления пустых и дублей: {len(df)}")
    return df

#базовая очистка текста от ссылок и мусора
def clean_data(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s\']', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

#лемматизация и удаление стоп-слов через spacy
def normalize_text(texts_series):
    results = []
    for doc in nlp.pipe(texts_series.tolist(), batch_size=1000, n_process=-1):
        lemmas = [
            token.lemma_.lower() if token.lemma_ else token.text.lower()
            for token in doc
            if not token.is_stop and not token.is_punct and not token.is_space and not token.is_digit and len(token) >= 2
        ]
        results.append(" ".join(lemmas))
    return results

def normalize_single_text(text):
    doc = nlp(text)
    lemmas = [
        token.lemma_.lower() if token.lemma_ else token.text.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space and not token.is_digit and len(token) >= 2
    ]
    return " ".join(lemmas)

#создание колонок с длиной текста и фильтрация коротких отзывов
def make_features(df):
    df['clean_raw_text'] = df['text'].apply(clean_data)
    df['clean_text'] = normalize_text(df['clean_raw_text'])
    
    df.replace("", np.nan, inplace=True)
    df.dropna(subset=['clean_text'], inplace=True)
    
    df['char_count'] = df['clean_raw_text'].str.len()
    df['token_count'] = df['clean_text'].str.split().str.len()
    
    df = df[df['token_count'] >= 3]
    save_to_log(f"После финальной фильтрации: {len(df)}")
    return df

def save_dataset(df, path):
    df.to_csv(path, index=False)
    print(f"Dataset saved to {path}")

def collect_stats(df):
    save_to_log("\n--- СТАТИСТИКА ---")
    plt.figure()
    df['rating'].value_counts().sort_index().plot(kind='bar', color='skyblue')
    plt.savefig(os.path.join(OUTPUT_DIR, 'rating_barplot.png'))
    plt.close()
    plt.figure()
    df['token_count'].plot(kind='hist', bins=50)
    plt.savefig(os.path.join(OUTPUT_DIR, 'token_histogram.png'))
    plt.close()
    save_to_log(f"Средняя длина (токены): {df['token_count'].mean()}")
    save_to_log(f"Средняя длина (символы): {df['char_count'].mean()}")
    save_to_log(f"Распределение:\n{df['rating'].value_counts().sort_index()}")

if __name__ == "__main__":
    save_to_log("\n" + "="*20 + "\nNEW RUN")
    df = load_data(INPUT_PATH)
    df = make_features(df)
    save_dataset(df, CSV_PATH)
    collect_stats(df)