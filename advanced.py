from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split

from sentiment_dataset import SentimentDataset

from utils import *

class AdvancedClassifier:
    def __init__(self, cfg, train = True, test = False) -> None:
        self.cfg = cfg
        if train and not test:
            self.df = read_data(cfg.data_preprocessed.path)
            self.heads = cfg.data_preprocessed.heads
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.advanced_model.model_name)

            self.load_data_train()

            self.model_pretrained = AutoModelForSequenceClassification.from_pretrained(
                                                                                        cfg.advanced_model.model_name, 
                                                                                        num_labels=cfg.advanced_model.num_labels
                                                                                    )
            self.model_trained = None
        else:
            None
        
    def load_data_train(self):
        cfg = self.cfg
        labels = self.df[self.heads[1]].values
        data = self.df[self.heads[0]].values
        train_txt, val_txt, train_labels, val_labels = train_test_split(data, labels, test_size=cfg.advanced_model.train_test_split)
        train_encodings = self.tokenizer(list(train_txt), truncation=True, padding=True)
        val_encodings = self.tokenizer(list(val_txt), truncation=True, padding=True)

        self.train_dataset = SentimentDataset(train_encodings, train_labels)
        self.eval_dataset = SentimentDataset(val_encodings, val_labels)
    
    def train(self) -> None:
        cfg = self.cfg
        training_args = TrainingArguments(
                                            output_dir=cfg.advanced_model.save_path, 
                                            num_train_epochs=cfg.advanced_model.num_train_epochs, 
                                            per_device_train_batch_size=cfg.advanced_model.per_device_train_batch_size,
                                            per_device_eval_batch_size=cfg.advanced_model.per_device_eval_batch_size,
                                            warmup_steps=cfg.advanced_model.warmup_steps,
                                            weight_decay=cfg.advanced_model.weight_decay,
                                            logging_dir=cfg.advanced_model.logging_dir,
                                            logging_steps=cfg.advanced_model.logging_steps
                                        )
        trainer = Trainer(
                            model=self.model_pretrained, 
                            args=training_args, 
                            train_dataset=self.train_dataset, 
                            eval_dataset=self.eval_dataset
                        )
        trainer.train()

    
    # Add inference function here
    # Inputs can be a prompt or from a file