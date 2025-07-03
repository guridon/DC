# train.py
from transformers import Trainer, TrainingArguments, set_seed
from dataset import ParagraphDataset
from model import get_model
from config import Config
import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}

def main():
    config=Config()
    set_seed(config.seed)

    
    train_data = ParagraphDataset("train").dataset
    val_data = ParagraphDataset("validation").dataset
    model = get_model()

    training_args = TrainingArguments(
        output_dir=Config.output_dir,
        evaluation_strategy="epoch",
        learning_rate=Config.learning_rate,
        per_device_train_batch_size=Config.batch_size,
        per_device_eval_batch_size=Config.batch_size,
        num_train_epochs=Config.num_epochs,
        weight_decay=Config.weight_decay,
        load_best_model_at_end=True,
        seed=Config.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,
        tokenizer=train_data.features["input_ids"].feature if hasattr(train_data, "features") else None,
    )

    trainer.train()

if __name__ == "__main__":
    main()



