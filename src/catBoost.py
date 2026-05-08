import os
import time
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from catboost import CatBoostClassifier, Pool

DATA_PATH = "../output/processed_reviews.csv"
BASE_ARTIFACTS_PATH = "../artifacts/catboost"

PATHS = {
    "report": os.path.join(BASE_ARTIFACTS_PATH, "classification_report.txt"),
    "matrix": os.path.join(BASE_ARTIFACTS_PATH, "confusion_matrix.png"),
    "model": os.path.join(BASE_ARTIFACTS_PATH, "catboost_model.pkl"),
    "info": os.path.join(BASE_ARTIFACTS_PATH, "catboost_info")
}

def setup_environment(path):
    os.makedirs(path, exist_ok=True)

#загрузка и подготовка данных для тренировки, тестов и валидации
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    
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
    
    train_pool = Pool(data=x_train, label=y_train, text_features=['text'])
    val_pool = Pool(data=x_val, label=y_val, text_features=['text'])
    
    return train_pool, val_pool, x_test, y_test

def train_model(train_pool, val_pool, info_dir):
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
        train_dir=info_dir,
        dictionaries=[{'dictionary_id': 'Word', 'gram_order': '3'}],
        feature_calcers=['BoW:top_tokens_count=10000', 'NaiveBayes'],
        tokenizers=[{'tokenizer_id': 'Space', 'separator_type': 'ByDelimiter', 'delimiter': ' '}]
    )
    
    start_time = time.time()
    model.fit(train_pool, eval_set=val_pool)
    duration = str(timedelta(seconds=int(time.time() - start_time)))
    
    return model, duration

def save_artifacts(model, x_test, y_test, duration, paths):
    test_pool = Pool(data=x_test, text_features=['text'])
    predictions = model.predict(test_pool)
    
    report = classification_report(y_test, predictions, digits=3)
    acc = accuracy_score(y_test, predictions)
    
    with open(paths["report"], "a", encoding="utf-8") as f:
        f.write(f"\n{'='*30}\n")
        f.write(f"=== Run: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write(report)
        f.write(f"\nAccuracy: {acc:.4f}\nTime: {duration}\n")
    
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.savefig(paths["matrix"], bbox_inches='tight')
    plt.close()
    
    joblib.dump(model, paths["model"])

if __name__ == "__main__":
    setup_environment(BASE_ARTIFACTS_PATH)
    
    train_pool, val_pool, x_test, y_test = load_and_prepare_data(DATA_PATH)
    model, duration = train_model(train_pool, val_pool, PATHS["info"])
    save_artifacts(model, x_test, y_test, duration, PATHS)