import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt 
import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import accuracy_score

x_train = joblib.load("../artifacts/x_train_tfidf.pkl")
y_train = joblib.load("../artifacts/y_train.pkl")
x_test = joblib.load("../artifacts/x_test_tfidf.pkl")
y_test = joblib.load("../artifacts/y_test.pkl")
vectorizer = joblib.load("../artifacts/tfidf_vectorizer.pkl")

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(x_train, y_train)

print("Model training completed")

predictions = model.predict(x_test)

report = classification_report(y_test, predictions, digits=3)
accuracy = accuracy_score(y_test, predictions)
report_path = "../artifacts/classification_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("=== Отчет о классификации моделей ===\n")
    f.write(report)
    f.write(f"\nAccuracy: {accuracy:.4f}\n")

cm = confusion_matrix(y_test, predictions)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)

plt.title('Матрица ошибок классификации рейтингов')
plt.xlabel('Предсказано')
plt.ylabel('Реальность')

plt.savefig("../artifacts/confusion_matrix.png", bbox_inches='tight')

feature_names = vectorizer.get_feature_names_out()

def write_top_words_to_file(file, class_label, n=10):
    class_index = list(model.classes_).index(class_label)
    coefs = model.coef_[class_index]
    
    top_indices = np.argsort(coefs)[-n:][::-1]
    bottom_indices = np.argsort(coefs)[:n]

    file.write(f"\n--- Топ-слова для рейтинга {class_label} ---\n")
    
    pos_words = [feature_names[i] for i in top_indices]
    file.write(f"ПОЗИТИВНЫЕ (тянут к {class_label}): " + ", ".join(pos_words) + "\n")
    
    neg_words = [feature_names[i] for i in bottom_indices]
    file.write(f"НЕГАТИВНЫЕ (отталкивают от {class_label}): " + ", ".join(neg_words) + "\n")

with open(report_path, "a", encoding="utf-8") as f:
    f.write("\n\n=== АНАЛИЗ ВЕСОВ МОДЕЛИ ===\n")
    write_top_words_to_file(f, 1)
    write_top_words_to_file(f, 5)

joblib.dump(model, "../artifacts/logreg_model.pkl")

print("script completed")