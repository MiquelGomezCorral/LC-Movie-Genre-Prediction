import torch
import tqdm
import joblib
import pandas as pd
import numpy 
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DebertaV2Tokenizer
)

from src.scripts import compute_metrics
from src.data import TransformerDataset

class TransformerMultiLabelTrainer:
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        tokenizer_name: str =  "distilbert-base-uncased",
        num_labels: int = 18,
    ):
        
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.num_labels = num_labels
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()
           
    def load_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True,
            problem_type="multi_label_classification"
        )
        
        model = model.to(self.device)
        
        return model
    
    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
        #tokenizer = DebertaV2Tokenizer.from_pretrained(self.model_name) 
        return tokenizer
    
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        
        predictions = torch.sigmoid(torch.tensor(predictions)).numpy()
        predictions_binary = (predictions > 0.5).astype(int)

        metrics = compute_metrics(labels, predictions_binary)

        return metrics

    def test(self, X_test):
        self.model.eval()

        mlb = joblib.load("models/binarizer.pkl")

        all_preds = []
        batch=8

        # Iterar en batches
        for i in tqdm.tqdm(range(0, len(X_test), batch), desc="Testing"):
            batch_texts = X_test[i:i + batch]

            # Tokenizar el batch
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                max_length=256,
                padding="max_length",
                return_tensors="pt"
            )

            # Mover inputs a dispositivo
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).int()

            all_preds.append(preds.cpu())

        # Concatenar todos los batches en un único tensor
        all_preds = torch.cat(all_preds, dim=0)
        y_pred = mlb.inverse_transform(all_preds.numpy())
        y_pred = [", ".join(pred) if pred else "" for pred in y_pred]
        return y_pred

    
    def train(self, train_dataset, val_dataset):
        train_dataset = TransformerDataset(dataset=train_dataset)
        val_dataset = TransformerDataset(dataset=val_dataset)
        train_dataset.tokenizer = self.tokenizer
        val_dataset.tokenizer = self.tokenizer
        
        training_args = TrainingArguments(
            output_dir= './results',
            num_train_epochs= 50,
            learning_rate=1e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_steps=100,
            eval_strategy='epoch',
            save_strategy="epoch",
            metric_for_best_model= "f1",
            load_best_model_at_end=True,
            greater_is_better=True,
            save_total_limit=2
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        trainer.train()

        metrics = trainer.state.log_history
        pd.DataFrame(metrics).to_csv("./results/metrics.csv", index=False)

        trainer.save_model("./results/best_model")   
        self.model.save_pretrained("./results/last_model")

        eval_results = trainer.evaluate()
