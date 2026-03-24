import os
import time
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

df = pd.read_csv("../output/processed_reviews.csv")

x_train_full, x_test, y_train_full, y_test = train_test_split(
    df[['text']], df['rating'],
    test_size=0.2,
    random_state=42,
    stratify=df['rating']
)

X_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full,
    test_size=0.2,
    random_state=42,
    stratify=y_train_full
)

vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=5,
    max_df=0.8,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train['text'])
X_val_tfidf = vectorizer.transform(x_val['text'])
X_test_tfidf = vectorizer.transform(x_test['text'])

joblib.dump(vectorizer, "../artifacts/tfidf_vectorizer_linear_svm.pkl")

model = LinearSVC(
    C=1.0,
    class_weight='balanced',
    random_state=42,
    max_iter=10000
)

start_time = time.time()

model.fit(X_train_tfidf, y_train)

end_time = time.time()
training_duration = end_time - start_time
formatted_time = str(timedelta(seconds=int(training_duration)))

print("Обучение модели завершено")


val_predictions = model.predict(X_val_tfidf)
test_predictions = model.predict(X_test_tfidf)


val_accuracy = accuracy_score(y_val, val_predictions)
val_macro_f1 = f1_score(y_val, val_predictions, average="macro", zero_division=0)
val_f1_class_1 = f1_score((y_val == 1).astype(int), (val_predictions == 1).astype(int), zero_division=0)
val_f1_class_5 = f1_score((y_val == 5).astype(int), (val_predictions == 5).astype(int), zero_division=0)


test_accuracy = accuracy_score(y_test, test_predictions)
test_macro_f1 = f1_score(y_test, test_predictions, average="macro", zero_division=0)
test_f1_class_1 = f1_score((y_test == 1).astype(int), (test_predictions == 1).astype(int), zero_division=0)
test_f1_class_5 = f1_score((y_test == 5).astype(int), (test_predictions == 5).astype(int), zero_division=0)

report = classification_report(y_test, test_predictions, digits=3)


report_path = "../artifacts/classification_report(LinearSVM).txt"

with open(report_path, "a", encoding="utf-8") as f:
    f.write("\n" + "="*30 + "\n")
    f.write("=== Новый запуск модели (Linear SVM) ===\n")

    f.write("\n--- Validation ---\n")
    f.write(f"Accuracy: {val_accuracy:.4f}\n")
    f.write(f"Macro F1: {val_macro_f1:.4f}\n")
    f.write(f"F1 (класс 1): {val_f1_class_1:.4f}\n")
    f.write(f"F1 (класс 5): {val_f1_class_5:.4f}\n")

    f.write("\n--- Test ---\n")
    f.write(f"Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Macro F1: {test_macro_f1:.4f}\n")
    f.write(f"F1 (класс 1): {test_f1_class_1:.4f}\n")
    f.write(f"F1 (класс 5): {test_f1_class_5:.4f}\n")

    f.write("\n=== Отчет о классификации ===\n")
    f.write(report)

    f.write(f"\nВремя обучения: {formatted_time}\n")


cm = confusion_matrix(y_test, test_predictions)

plt.figure()
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=sorted(df['rating'].unique()),
    yticklabels=sorted(df['rating'].unique())
)

plt.title('Матрица ошибок классификации рейтингов (Linear SVM)')
plt.xlabel('Предсказано')
plt.ylabel('Реальность')

plt.savefig("../artifacts/confusion_matrix(LinearSVM).png", bbox_inches='tight')

joblib.dump(model, "../artifacts/linear_svm_model.pkl")
