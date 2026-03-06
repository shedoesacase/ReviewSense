import pandas as pd
import spacy
import matplotlib.pyplot as plt
import numpy as np
import re
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_to_log(message, filename="stats.txt"):
    log_path = os.path.join(OUTPUT_DIR, filename)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(str(message) + "\n")

save_to_log("\n______NEW RUN_______")

originalDf = pd.read_json("../data/All_Beauty.jsonl", lines=True)
save_to_log(f"Оригинальное количество строк: {originalDf.shape[0]}")

Rating_Text = originalDf[["rating", "text"]].copy()
Rating_Text = Rating_Text.dropna()
Rating_Text.drop_duplicates(keep='first', inplace=True)

save_to_log(f"После удаления пустых строк и дубликатов: {Rating_Text.shape[0]}")

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

#очистка текста
def clean_raw_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s\']', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

#Приведение к нижнему регистру и токенизация
def tokenize_and_lemmatize(texts):
    results = []
    for doc in nlp.pipe(texts.tolist(), batch_size=1000, n_process=-1):
        lemmas = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct and not token.is_space and not token.is_digit and len(token) >= 2
        ]
        results.append(" ".join(lemmas))
    return results

#финальная очистка
def final_normalize(df):
    df['clean_raw_text'] = df['text'].apply(clean_raw_text)
    df['clean_text'] = tokenize_and_lemmatize(df['clean_raw_text'])
    df['clean_text'] = df['clean_text'].str.strip()
    df.replace("", np.nan, inplace=True)
    df.dropna(subset=['clean_text'], inplace=True)
    df['char_count'] = df['clean_raw_text'].str.len()
    df['token_count'] = df['clean_text'].str.split().str.len()
    df = df[df['token_count'] >= 3]
    save_to_log(f"После финальной фильтрации: {df.shape[0]}")
    csv_path = os.path.join(OUTPUT_DIR, "processed_reviews.csv")
    df.to_csv(csv_path, index=False)
    print("finish f normalize")
    return df

#статистика
def stats(df):
    print("start collecting stats")
    plt.figure()
    ax = df['rating'].value_counts().sort_index().plot(kind='bar', color='skyblue', rot=0)
    ax.set_xlabel("Рейтинг")
    ax.set_ylabel("Количество строк")
    plt.savefig(os.path.join(OUTPUT_DIR, 'rating_barplot.png'), dpi=300)
    plt.close()

    save_to_log("Распределение рейтингов:")
    save_to_log(df['rating'].value_counts().sort_index().to_string())

    plt.figure()
    df['token_count'].plot(kind='hist', bins=50)
    plt.xlabel("Длина отзыва (в токенах)")
    plt.ylabel("Количество")
    plt.savefig(os.path.join(OUTPUT_DIR, 'token_histogram.png'), dpi=300)
    plt.close()

    avg_count = df['token_count'].mean()
    max_count = df['token_count'].max()
    min_count = df['token_count'].min()

    save_to_log(f"Средняя длина: {avg_count}")
    save_to_log(f"Максимальная длина: {max_count}")
    save_to_log(f"Минимальная длина: {min_count}")

    avg_chars = df['char_count'].mean()
    save_to_log(f"Средняя длина в символах: {avg_chars}")

    save_to_log(f"5 случайных строк:")
    save_to_log(df.sample(5).to_string())
    print("finish collecting stats")

if __name__ == "__main__":
    df = final_normalize(Rating_Text)
    stats(df)
    print("script complete")


#TODO переделать структуру файла под более нормальную: 
#load_data()
#clean_data()
#normalize_text()
#make_features()
#save_dataset()
#stats()