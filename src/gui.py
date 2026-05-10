import customtkinter as ctk
from preprocessing import clean_data, normalize_single_text
import joblib
from transformers import pipeline
import threading
import pandas as pd
from catboost import  Pool


class ReviewSenseApp:
    def __init__(self, root, predicter):
        self.root = root
        self.predicter = predicter
        root.geometry("1000x600")
        root.title("ReviewSense")
        self.names = ["LogReg", "LinearSVM", "CatBoost", "RoBERTa"]
        self.result_labels = []
        self.progress_bars = []
        self.percent_labels = []

        self.setup_ui()

    def setup_ui(self):
        self.entry = ctk.CTkTextbox(root, width=400, height=450, border_width=5)
        self.entry.pack(side="top", anchor="nw", padx=(50, 0), pady=(50, 0))

        self.btn = ctk.CTkButton(master=root, text="Send", width=400, height=40, command=self.send_message)
        self.btn.pack(side="top", anchor="nw", padx=(50, 0), pady=(10, 0))
        self.entry.bind("<Return>", self.send_message)
        self.entry.bind("<Shift-Return>", self.handle_shift_enter)

        msg = "Приложение работает только с английским языком. The application works only with English language."

        self.bottom_label = ctk.CTkLabel(root, text=msg, text_color="gray")
        self.bottom_label.pack(side="bottom")

        self.rating_panel = ctk.CTkFrame(root, fg_color="transparent")
        self.rating_panel.place(relx=0.55, rely=0.1)

        self.probability_title = ctk.CTkLabel(self.rating_panel, text="Model Confidence / Probability", font=("Arial", 26, "bold"), text_color="#1f6aa5")
        self.probability_title.grid(row=0, column=0, columnspan=1, pady=(0, 0), sticky="w")

        for i, name in enumerate(self.names):
            self.create_model_row(i, name)

    def create_model_row(self, index, name):
        item_frame = ctk.CTkFrame(self.rating_panel, fg_color="transparent")
        item_frame.grid(row=index+1, column=0, pady=10, padx=10, sticky="w")
        
        lbl_title = ctk.CTkLabel(item_frame, text=name, font=("Arial", 16, "bold"), text_color="gray")
        lbl_title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 5))

        lbl_symbol = ctk.CTkLabel(item_frame, text="0", width=60, height=60,fg_color="#3b3b3b", text_color="white", corner_radius=10, font=("Arial", 28, "bold"))
        lbl_symbol.grid(row=1, column=0)
        self.result_labels.append(lbl_symbol)

        if index == 1:
            warning_msg = ctk.CTkLabel(item_frame, text="LinearSVM has no probability", font=("Arial", 20, "italic"), text_color="gray")
            warning_msg.grid(row=1, column=1, padx=15, sticky="w")
            self.percent_labels.append(None)
            self.progress_bars.append(None)
        else:
        
            pb = ctk.CTkProgressBar(item_frame, width=250, height=25, corner_radius=5)
            pb.set(0.65)
            pb.grid(row=1, column=1, padx=15, sticky="ew")
            self.progress_bars.append(pb)

            pb_text = ctk.CTkLabel(pb, text="65%", font=("Arial", 12, "bold"), text_color="white", fg_color="transparent", width=0, height=0, pady=0, padx=0)
            pb_text.place(relx=0.5, rely=0.5, anchor="center")
            self.percent_labels.append(pb_text)
    
    def send_message(self, event=None):
        message = self.entry.get("0.0", "end-1c")
        if message:
            print(f"Сообщение отправлено: {message}")
            self.btn.configure(state="disabled", text="processing...")
            thread = threading.Thread(target=self.run_inference, args=(message,), daemon=True)
            thread.start()
        return "break"
    
    def run_inference(self, message):
        cleaned_text = clean_data(message)
        deep_text = normalize_single_text(cleaned_text)

        results = self.predicter.predict(cleaned_text, deep_text)
        
        self.root.after(0, self.update_res, results)
    
    def handle_shift_enter(self, event):
        pass
    
    def update_res(self, results):
        keys = ["logreg", "svm", "catboost", "roberta"]

        for i, key in enumerate(keys):
            data = results[key]
            self.result_labels[i].configure(text=str(data["prediction"]))

            if self.progress_bars[i] and data["probability"] is not None:
                prob = data["probability"]
                if isinstance(prob, list):
                    prob_value = max(prob)
                else:
                    prob_value = prob

                self.progress_bars[i].set(prob_value)
                self.percent_labels[i].configure(text=f"{int(prob_value*100)}%")
        self.btn.configure(state="normal", text="Send")




class Predicter:
    def __init__(self):
        BASE_ARTIFACTS_PATH = "../artifacts"
        LOGREG_PATH = "/logreg"
        SVM_PATH = "/linear_svm"
        CATBOOST_PATH = "/catboost"
        self.logreg_tfidf = joblib.load(f"{BASE_ARTIFACTS_PATH}/tfidf_vectorizer.pkl")
        self.svm_tfidf = joblib.load(f"{BASE_ARTIFACTS_PATH}{SVM_PATH}/tfidf_vectorizer.pkl")
        self.model_logreg = joblib.load(f"{BASE_ARTIFACTS_PATH}{LOGREG_PATH}/logreg_model.pkl")
        self.model_svm = joblib.load(f"{BASE_ARTIFACTS_PATH}{SVM_PATH}/linear_svm_model.pkl")
        self.model_catboost = joblib.load(f"{BASE_ARTIFACTS_PATH}{CATBOOST_PATH}/catboost_model.pkl")
        self.pipe = pipeline("text-classification", model="../artifacts/roberta_model", tokenizer="roberta-base", device=0)

    def predict(self, light_text, clean_text):
        tfidf_features_logreg = self.logreg_tfidf.transform([clean_text])
        tfidf_features_svm = self.svm_tfidf.transform([clean_text])
        logreg_prediction = self.model_logreg.predict(tfidf_features_logreg)
        logreg_probs = self.model_logreg.predict_proba(tfidf_features_logreg)
        svm_prediction = self.model_svm.predict(tfidf_features_svm)
        
        cat_data = pd.DataFrame([light_text], columns=['text'])
        predict_pool = Pool(data=cat_data, text_features=['text'])
        
        catboost_prediction = self.model_catboost.predict(predict_pool)
        catboost_probs = self.model_catboost.predict_proba(predict_pool)
        roberta_prediction = self.pipe(light_text)
        res_roberta = roberta_prediction[0]

        output = {
            "logreg": {
                "prediction": int(logreg_prediction[0]),
                "probability": logreg_probs[0].tolist()
            },
            "svm": {
                "prediction": int(svm_prediction[0]),
                "probability": None
            },
            "catboost": {
                "prediction": int(catboost_prediction.flatten()[0]),
                "probability": catboost_probs[0].tolist()
            },
            "roberta": {
                "prediction": int(res_roberta["label"].split('_')[-1]),
                "probability": res_roberta["score"]
            },
        }

        return output









if __name__ == "__main__":
    root_loading = ctk.CTk()
    root_loading.title("Загрузка")
    root_loading.geometry("300x150")
    
    label = ctk.CTkLabel(root_loading, text="Загрузка моделей...\nПожалуйста, подождите")
    label.pack(expand=True)
    
    root_loading.update() 
    predicter = Predicter()
    root_loading.destroy()
    root = ctk.CTk()
    app = ReviewSenseApp(root, predicter)
    root.mainloop()

