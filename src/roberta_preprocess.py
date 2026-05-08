import pandas as pd
import re
from datasets import Dataset, ClassLabel
from transformers import AutoTokenizer


def preprocess(dataPath): #удалили пустые строки и повторки, привели все к строковому типу, обернули в лучше воспринимаемый библиотекой тип данных
    originalDf = pd.read_json(dataPath, lines=True)
    Rating_Text = originalDf[["rating", "text"]].copy()
    Rating_Text = Rating_Text.dropna()
    Rating_Text.drop_duplicates(keep='first', inplace=True)
    Rating_Text['rating'] = pd.to_numeric(Rating_Text['rating'], errors='coerce').dropna().astype(int)
    Rating_Text['rating'] = Rating_Text['rating'] - 1  # Делаем 0-4
    Rating_Text['text'] = Rating_Text['text'].astype(str).apply(clean_text)
    counts = Rating_Text['rating'].value_counts()
    min_size = counts.min()
    balanced_df = pd.concat([
        Rating_Text[Rating_Text['rating'] == label].sample(min_size, random_state=42) 
        for label in Rating_Text['rating'].unique()
    ])
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    dataset = Dataset.from_pandas(balanced_df)
    if "__index_level_0__" in dataset.column_names:
        dataset = dataset.remove_columns(["__index_level_0__"])
    dataset = dataset.cast_column("rating", ClassLabel(num_classes=5, names=["1","2","3","4","5"]))
    dataset = dataset.rename_column("rating", "labels")

    return dataset

def trainTestSplit(dataset):
    return dataset.train_test_split(test_size = 0.2, seed = 42)

def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize(batch):
    encoded = tokenizer(
        batch["text"],
        truncation = True,
        max_length = 512
    )
    return encoded

if __name__ == "__main__":
    dataPath = "../data/All_Beauty.jsonl"
    dataset = preprocess(dataPath)
    dataset = dataset.shuffle(seed=42).select(range(50000))
    splitted_dataset = trainTestSplit(dataset)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    tokenized_dataset = splitted_dataset.map(tokenize, batched=True, batch_size=1000, remove_columns="text")
    tokenized_dataset.save_to_disk("tokenized_dataset")
    