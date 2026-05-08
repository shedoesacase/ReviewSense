import os
import time
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "output", "processed_reviews.csv")
SAVE_DIR = os.path.join(BASE_DIR, "..", "artifacts", "linear_svm")

def setup_dirs(path):
    os.makedirs(path, exist_ok=True)

#загрузка данных и разделение
def load_data(path):
    df = pd.read_csv(path)
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        df[['text']], df['rating'],
        test_size=0.2,
        random_state=42,
        stratify=df['rating']
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full
    )
    return x_train, x_val, x_test, y_train, y_val, y_test

#обучение векторизатора и модели
def train_svm(x_train, y_train):
    vectorizer = TfidfVectorizer(
        max_features=20000,
        min_df=5,
        max_df=0.85,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    
    start_time = time.time()
    x_train_tfidf = vectorizer.fit_transform(x_train['text'])
    
    model = LinearSVC(
        C=0.04,
        class_weight='balanced',
        max_iter=20000,
        random_state=42
    )
    model.fit(x_train_tfidf, y_train)
    
    duration = str(timedelta(seconds=int(time.time() - start_time)))
    return model, vectorizer, duration

#оценка и сохранение результатов
def evaluate_and_save(model, vectorizer, x_val, y_val, x_test, y_test, duration, save_path):
    x_val_tfidf = vectorizer.transform(x_val['text'])
    x_test_tfidf = vectorizer.transform(x_test['text'])
    
    val_preds = model.predict(x_val_tfidf)
    test_preds = model.predict(x_test_tfidf)
    
    metrics = {
        "val_acc": accuracy_score(y_val, val_preds),
        "test_acc": accuracy_score(y_test, test_preds),
        "test_f1": f1_score(y_test, test_preds, average="macro", zero_division=0),
        "report": classification_report(y_test, test_preds, digits=3)
    }
    
    report_file = os.path.join(save_path, "classification_report.txt")
    with open(report_file, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*30}\nRun: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Val Accuracy: {metrics['val_acc']:.4f}\nTest Accuracy: {metrics['test_acc']:.4f}\n")
        f.write(f"Test Macro F1: {metrics['test_f1']:.4f}\nDuration: {duration}\n")
        f.write(f"\nDetailed Report:\n{metrics['report']}\n")
    
    cm = confusion_matrix(y_test, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.savefig(os.path.join(save_path, "confusion_matrix.png"))
    plt.close()
    
    joblib.dump(vectorizer, os.path.join(save_path, "tfidf_vectorizer.pkl"))
    joblib.dump(model, os.path.join(save_path, "linear_svm_model.pkl"))

if __name__ == "__main__":
    setup_dirs(SAVE_DIR)
    x_train, x_val, x_test, y_train, y_val, y_test = load_data(DATA_PATH)
    model, vectorizer, duration = train_svm(x_train, y_train)
    evaluate_and_save(model, vectorizer, x_val, y_val, x_test, y_test, duration, SAVE_DIR)