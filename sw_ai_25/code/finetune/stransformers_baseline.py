import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, Trainer, TrainingArguments, PreTrainedModel, PretrainedConfig, AutoModel
)
from sentence_transformers import models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    average_precision_score, matthews_corrcoef, confusion_matrix, roc_curve
)
import wandb

# config
NAME="qwen_filtered"
DATA_PATH  = "../../../data/new_qwen/train_electra_generated_paragraph_qwen32b_filtered.csv"
MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
SEED       = 42
# KEY = "title"
TEXT_COL   = "paragraph_text"
LABEL_COL  = "labels"
BATCH_SIZE = 32  
NUM_EPOCHS = 1
MAX_LENGTH = 512
LEARNING_RATE = 5e-5
# MAX_PARAGRAPHS = 100

wandb.init(
    project="ai-fake-text-detection",
    name=f"koelectra-v3-{NAME}",
    config={
        "model_name": MODEL_NAME,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "max_length": MAX_LENGTH,
        "learning_rate": LEARNING_RATE,
    }
)

df_raw = pd.read_csv(DATA_PATH)
df = df_raw[[TEXT_COL, "generated"]].rename(columns={"generated": LABEL_COL}).dropna().reset_index(drop=True)
df[LABEL_COL] = df[LABEL_COL].astype(int)

print("🔹 학습용 문장 수:", len(df))
print("🔹 라벨 분포:")
print(df[LABEL_COL].value_counts())

train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df[LABEL_COL])

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Dataset
class ParagraphDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.texts = df[TEXT_COL].tolist()
        self.labels = df[LABEL_COL].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_dataset = ParagraphDataset(train_df, tokenizer, MAX_LENGTH)
val_dataset = ParagraphDataset(val_df, tokenizer, MAX_LENGTH)

# mean pooling + binary classifier
class KoELECTRAMeanPoolingClassifier(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (batch, seq, hidden)
        # mean pooling 
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        masked = last_hidden * mask
        summed = torch.sum(masked, 1)
        summed_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / summed_mask
        logits = self.classifier(mean_pooled)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}

# compute_metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    preds = np.round(probs).astype(int)
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    pr_auc = average_precision_score(labels, probs)
    mcc = matthews_corrcoef(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr, tpr, thresholds = roc_curve(labels, probs)
    wandb.log({
        "val/accuracy": acc,
        "val/roc_auc": auc,
        "val/precision": precision,
        "val/recall": recall,
        "val/f1": f1,
        "val/pr_auc": pr_auc,
        "val/specificity": specificity,
        "val/mcc": mcc,
        "val/roc_curve_fpr": fpr.tolist(),
        "val/roc_curve_tpr": tpr.tolist(),
        "val/roc_curve_thresholds": thresholds.tolist(),
    })
    return {
        "accuracy": acc,
        "roc_auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "specificity": specificity,
        "mcc": mcc,
    }

# TrainingArguments 
training_args = TrainingArguments(
    output_dir="./checkpoint",
    seed=SEED,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    # save_steps=3000,
    # eval_steps=3000,
    load_best_model_at_end=True,
    metric_for_best_model="roc_auc",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    report_to=["wandb"],
)

# model & trainer
model = KoELECTRAMeanPoolingClassifier(MODEL_NAME, num_labels=2)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ----------------- 학습 및 저장 -----------------
trainer.train()
trainer.save_model("./checkpoint_best")
tokenizer.save_pretrained("./checkpoint_best")
wandb.finish()
print("✅ 모델 저장 → ./checkpoint_best")
