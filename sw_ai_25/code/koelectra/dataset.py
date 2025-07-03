# dataset.py
from datasets import load_dataset
from transformers import AutoTokenizer
from config import Config

class ParagraphDataset:
    def __init__(self, split):
        self.tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
        files = {"train": Config.train_file, "validation": Config.val_file}
        self.dataset = load_dataset("csv", data_files={split: files[split]})[split]
        self.dataset = self.dataset.map(self.preprocess, batched=True)

    def preprocess(self, examples):
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=Config.max_length,
        )
