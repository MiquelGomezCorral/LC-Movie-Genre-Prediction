import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

from src.scripts import compute_metrics
from src.data import TransformerDataset

class TransformerMultiLabelTrainer:
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 18,
    ):
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()
           
    def load_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type="multi_label_classification"
        )
        
        model = model.to(self.device)
        
        return model
    
    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        
        return tokenizer
    
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        
        predictions = torch.sigmoid(torch.tensor(predictions)).numpy()
        predictions_binary = (predictions > 0.5).astype(int)

        metrics = compute_metrics(labels, predictions_binary)

        return metrics

    
    def train(self, train_dataset, val_dataset):
        train_dataset = TransformerDataset(dataset=train_dataset)
        val_dataset = TransformerDataset(dataset=val_dataset)
        train_dataset.tokenizer = self.tokenizer
        val_dataset.tokenizer = self.tokenizer
        
        training_args = TrainingArguments(
            output_dir= './results',
            num_train_epochs= 5,
            learning_rate=5e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            eval_strategy='epoch',
            save_strategy="epoch",
            metric_for_best_model= "f1",
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        result = trainer.train()

        eval_results = trainer.evaluate()