import os
import torch
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
)
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.functional import softmax
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    average_precision_score, matthews_corrcoef, confusion_matrix, roc_curve
)
import wandb

# config
DATA_PATH  = "./finetune/train_paragraph_rewrite_v1_batch1.csv"
MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
SEED       = 42
KEY = "title"
TEXT_COL   = "paragraph_text"
LABEL_COL  = "labels"
BATCH_SIZE = 2 
NUM_EPOCHS = 6
MAX_LENGTH = 128
LEARNING_RATE = 5e-5
MAX_PARAGRAPHS = 30

# wandb logging
wandb.init(
    project="ai-fake-text-detection",
    name=f"koelectra-v3-mil-balanced",
    config={
        "model_name": MODEL_NAME,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "max_length": MAX_LENGTH,
        "learning_rate": LEARNING_RATE,
    }
)

# file_path
output_dir = "../data/finetune/"
files = sorted([
    fname for fname in os.listdir(output_dir)
    if fname.startswith("train_paragraph_rewrite_v1") and fname.endswith(".csv")
])

# df_list = []
# for file in files:
#     file_path = os.path.join(output_dir, file)
#     df = pd.read_csv(file_path, usecols=["paragraph_text", "generated", "title"]) 
#     df_list.append(df)

# train_csv = pd.concat(df_list, ignore_index=True)
# len(train_csv)

df_raw = pd.read_csv(DATA_PATH)
df = df_raw[[TEXT_COL, "generated", KEY]].copy()
df = df.rename(columns={"generated": LABEL_COL})
df = df.dropna().sample(frac=1, random_state=SEED).reset_index(drop=True)

if "doc_id" not in df.columns:
    df["doc_id"] = df[KEY].astype("category").cat.codes

print("🔹 학습용 문장 수:", len(df))
print("🔹 라벨 분포:")
print(df[LABEL_COL].value_counts())

# split train_val
train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df[LABEL_COL]
)

# tokenizer 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = "[PAD]"
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 문서 단위 Dataset
class DocumentBatchDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=MAX_LENGTH, max_paragraphs=MAX_PARAGRAPHS):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_paragraphs = max_paragraphs
        self.groups = []
        self.labels = []
        for label in [0, 1]:
            for _, group in df[df[LABEL_COL] == label].groupby(KEY):
                if len(group) > max_paragraphs:
                    group = group.iloc[:max_paragraphs]
                self.groups.append(group)
                self.labels.append(label)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]
        batch = []
        for _, item in group.iterrows():
            encoding = self.tokenizer(
                item[KEY],
                item[TEXT_COL],
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors="pt"
            )
            out = {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "doc_id": torch.tensor(int(item["doc_id"]), dtype=torch.long),
                "labels": torch.tensor(int(item[LABEL_COL]), dtype=torch.long),
            }
            batch.append(out)
        return batch

# 문단 단위 flatten
def collate_fn(batch):
    batch = [item for group in batch for item in group]
    all_keys = set().union(*(x.keys() for x in batch))
    out = {}
    for key in all_keys:
        vals = [x[key] for x in batch]
        if isinstance(vals[0], torch.Tensor):
            out[key] = torch.stack(vals)
        else:
            out[key] = torch.tensor(vals)
    return out

# Balanced Batch Sampler
class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        assert batch_size % 2 == 0
        self.class0_idx = np.where(self.labels == 0)[0]
        self.class1_idx = np.where(self.labels == 1)[0]
        self.num_class0 = len(self.class0_idx)
        self.num_class1 = len(self.class1_idx)
        self.num_batches = (self.num_class0 + self.num_class1) // batch_size

    def __iter__(self):
        np.random.shuffle(self.class0_idx)
        np.random.shuffle(self.class1_idx)
        ptr0, ptr1 = 0, 0
        for _ in range(self.num_batches):
            batch = []
            half_batch = self.batch_size // 2
            take0 = min(half_batch, self.num_class0 - ptr0)
            take1 = min(half_batch, self.num_class1 - ptr1)
            if take0 < half_batch:
                take1 = min(self.batch_size - take0, self.num_class1 - ptr1)
            if take1 < half_batch:
                take0 = min(self.batch_size - take1, self.num_class0 - ptr0)
            batch.extend(self.class0_idx[ptr0:ptr0+take0])
            batch.extend(self.class1_idx[ptr1:ptr1+take1])
            ptr0 += take0
            ptr1 += take1
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches

# dataset 
train_dataset = DocumentBatchDataset(train_df, tokenizer, max_length=MAX_LENGTH)
val_dataset = DocumentBatchDataset(val_df, tokenizer, max_length=MAX_LENGTH)

# sampler 
train_sampler = BalancedBatchSampler(train_dataset.labels, batch_size=BATCH_SIZE)
val_sampler = BalancedBatchSampler(val_dataset.labels, batch_size=BATCH_SIZE)

# 모델 
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "ORIGINAL", 1: "GENERATED"},
    label2id={"ORIGINAL": 0, "GENERATED": 1},
)
model.config.pad_token_id = tokenizer.pad_token_id

# MIL Trainer 
class MILTrainer(Trainer):
    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=0,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return DataLoader(
            eval_dataset,
            batch_sampler=val_sampler,
            collate_fn=collate_fn,
            num_workers=0,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        doc_ids = inputs.pop("doc_id")
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[:, 1]
        doc_ids_np = doc_ids.cpu().numpy()
        labels_np = labels.cpu().numpy()
        doc2probs = {}
        doc2label = {}
        for i, doc_id in enumerate(doc_ids_np):
            doc2probs.setdefault(doc_id, []).append(probs[i])
            doc2label[doc_id] = labels_np[i]
        group_probs = torch.stack([torch.max(torch.stack(doc2probs[doc_id])) for doc_id in doc2probs])
        group_labels = torch.tensor([doc2label[doc_id] for doc_id in doc2probs], device=group_probs.device)
        # Label Smoothing
        # smoothing = 0.1 
        # group_labels = group_labels.float() * (1 - smoothing) + 0.5 * smoothing
        loss = torch.nn.functional.binary_cross_entropy(group_probs, group_labels.float())
        return (loss, outputs) if return_outputs else loss

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
        "roc_curve_fpr": fpr.tolist(),
        "roc_curve_tpr": tpr.tolist(),
        "roc_curve_thresholds": thresholds.tolist()
    })

    result = {
        "accuracy": acc,
        "roc_auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "specificity": specificity,
        "mcc": mcc,
    }
    wandb.log(result)
    return result

# TrainingArguments 
training_args = TrainingArguments(
    output_dir="./checkpoint",
    seed=SEED,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=5000,
    eval_steps=5000,
    load_best_model_at_end=True,
    metric_for_best_model="roc_auc",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    report_to=["wandb"],
)

# Trainer 
trainer = MILTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

trainer.train()
trainer.save_model("./checkpoint_best")
tokenizer.save_pretrained("./checkpoint_best")
print("✅ 모델 && 토크나이저 저장 → ./checkpoint_best")

wandb.finish()
