# model.py
from transformers import AutoModelForSequenceClassification
from config import Config

def get_model():
    return AutoModelForSequenceClassification.from_pretrained(
        Config.model_name,
        num_labels=Config.num_labels
    )
