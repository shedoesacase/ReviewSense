import os
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "..", "artifacts", "logreg")

def setup_env(path):
    os.makedirs(path, exist_ok=True)

#загрузка готовых признаков
def load_processed_data():
    x_train = joblib.load(os.path.join(BASE_DIR, "..", "artifacts", "x_train_tfidf.pkl"))
    y_train = joblib.load(os.path.join(BASE_DIR, "..", "artifacts", "y_train.pkl"))
    x_test = joblib.load(os.path.join(BASE_DIR, "..", "artifacts", "x_test_tfidf.pkl"))
    y_test = joblib.load(os.path.join(BASE_DIR, "..", "artifacts", "y_test.pkl"))
    vectorizer = joblib.load(os.path.join(BASE_DIR, "..", "artifacts", "tfidf_vectorizer.pkl"))
    return x_train, y_train, x_test, y_test, vectorizer

def train_logreg(x, y):
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(x, y)
    return model

#анализ важных слов для классов
def write_weights(file, model, vectorizer, class_label, n=10):
    feature_names = vectorizer.get_feature_names_out()
    class_index = list(model.classes_).index(class_label)
    coefs = model.coef_[class_index]
    top_indices = np.argsort(coefs)[-n:][::-1]
    bottom_indices = np.argsort(coefs)[:n]
    file.write(f"\n--- топ-слова для рейтинга {class_label} ---\n")
    pos_words = [feature_names[i] for i in top_indices]
    file.write(f"позитивные: " + ", ".join(pos_words) + "\n")
    neg_words = [feature_names[i] for i in bottom_indices]
    file.write(f"негативные: " + ", ".join(neg_words) + "\n")

#сохранение всех отчетов и графиков
def save_artifacts(model, x_test, y_test, vectorizer, path):
    preds = model.predict(x_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, digits=3)
    report_path = os.path.join(path, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== отчет о классификации (logreg) ===\n")
        f.write(report)
        f.write(f"\naccuracy: {acc:.4f}\n")
        f.write("\n\n=== анализ весов модели ===\n")
        write_weights(f, model, vectorizer, 1)
        write_weights(f, model, vectorizer, 5)
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('матрица ошибок (logreg)')
    plt.savefig(os.path.join(path, "confusion_matrix.png"), bbox_inches='tight')
    plt.close()
    joblib.dump(model, os.path.join(path, "logreg_model.pkl"))

if __name__ == "__main__":
    setup_env(ARTIFACTS_DIR)
    x_tr, y_tr, x_ts, y_ts, vect = load_processed_data()
    clf = train_logreg(x_tr, y_tr)
    save_artifacts(clf, x_ts, y_ts, vect, ARTIFACTS_DIR)
    print("logreg обучение завершено")