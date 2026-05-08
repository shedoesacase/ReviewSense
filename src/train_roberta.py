import numpy as np
from datasets import load_from_disk
from transformers import TrainingArguments, DataCollatorWithPadding, Trainer, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score

def modelLoad():
    id2label = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}
    label2id = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
    config = AutoConfig.from_pretrained('roberta-base')
    model = AutoModelForSequenceClassification.from_pretrained(
    'roberta-base', 
    num_labels=5, 
    id2label=id2label, 
    label2id=label2id
    )
    return model

def computeMetrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, predictions)
    f1_per_class = f1_score(labels, predictions, average=None)
    f1_macro = np.mean(f1_per_class)
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_class_1": f1_per_class[0],
        "f1_class_5": f1_per_class[4]
    }
    

def getTrainingArgs():
    args = TrainingArguments(
        output_dir="./results",
        learning_rate=1e-5,
        per_device_eval_batch_size=32,
        per_device_train_batch_size=32,
        num_train_epochs=4,
        weight_decay=0.1,
        warmup_steps=500,
        fp16=True,
        logging_dir="./logs",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro"
    )
    return args

if __name__ == "__main__":
    dataPath = "./tokenized_dataset/"
    dataset = load_from_disk(dataPath)
    model = modelLoad()
    args = getTrainingArgs()
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(model=model, args=args, train_dataset=dataset["train"], eval_dataset=dataset["test"], compute_metrics=computeMetrics, data_collator=data_collator)
    train_result = trainer.train()
    total_train_time = train_result.metrics["train_runtime"]
    print(f"Время обучения: {total_train_time:.2f} сек. ({total_train_time/60:.2f} мин.)")
    final_metrics = trainer.evaluate()
    print(f"Accuracy: {final_metrics['eval_accuracy']:.4f}")
    print(f"F1 (1 звезда): {final_metrics['eval_f1_class_1']:.4f}")
    print(f"F1 (5 звезд): {final_metrics['eval_f1_class_5']:.4f}")
    print(f"Macro F1: {final_metrics['eval_f1_macro']:.4f}")
    trainer.save_model("../artifacts/roberta_model")
    
    
