import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import os
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 
import seaborn as sns
from datetime import timedelta
import time

df = pd.read_csv("../output/processed_reviews.csv")

x_train_full, x_test, y_train_full, y_test = train_test_split(df[['text']], df['rating'], test_size=0.2, random_state=42, stratify=df['rating'])

X_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)

train_data = Pool(data=X_train, label=y_train, text_features=['text'])

val_data = Pool(data=x_val, label=y_val, text_features=['text'])

model = CatBoostClassifier(
    iterations=2500, 
    learning_rate=0.03, 
    loss_function='MultiClass', 
    random_seed=42, 
    task_type='GPU',
    depth=8,
    auto_class_weights='Balanced',
    early_stopping_rounds=100,
    verbose=100,
    l2_leaf_reg=2,
    dictionaries=[{
        'dictionary_id': 'Word',
        'gram_order': '3'
    }],
    feature_calcers = [
        'BoW:top_tokens_count=10000',
        'NaiveBayes'
    ],
    tokenizers=[{
        'tokenizer_id': 'Space',
        'separator_type': 'ByDelimiter',
        'delimiter': ' '
    }],
    )

start_time = time.time()

model.fit(train_data, eval_set=val_data)

end_time = time.time()
training_duration = end_time - start_time
formatted_time = str(timedelta(seconds=int(training_duration)))

print("Model training completed")

test_pool = Pool(data=x_test, text_features=['text'])
predictions = model.predict(test_pool)

report = classification_report(y_test, predictions, digits=3)
accuracy = accuracy_score(y_test, predictions)
report_path = "../artifacts/classification_report(CatBoost_Classifier).txt"
with open(report_path, "a", encoding="utf-8") as f:
    f.write("\n" + "="*30 + "\n")
    f.write("=== Новый запуск модели (CatBoost с лучшими параметрами) ===\n")
    f.write("=== Отчет о классификации моделей ===\n")
    f.write(report)
    f.write(f"\nAccuracy: {accuracy:.4f}\n")
    f.write(f"Время обучения: {formatted_time}\n")

cm = confusion_matrix(y_test, predictions)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)

plt.title('Матрица ошибок классификации рейтингов')
plt.xlabel('Предсказано')
plt.ylabel('Реальность')

plt.savefig("../artifacts/confusion_matrix(catBoost_Classifier).png", bbox_inches='tight')

joblib.dump(model, "../artifacts/catBoost_Classifier_model.pkl")