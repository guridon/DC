import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import torch.nn.functional as F

# config
num=6000
checkpoint_dir = f"./checkpoint/checkpoint-{num}" 
max_length = 512
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestParagraphDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.data = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.loc[idx]
        title = item["title"]
        paragraph_text = item["paragraph_text"]
        encoding = self.tokenizer(
            title,
            paragraph_text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        out = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }
        return out

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
model.to(device)

# test dataset
test_df = pd.read_csv('../../data/test.csv')
test_dataset = TestParagraphDataset(test_df, tokenizer, max_length=max_length)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def predict_proba(model, loader, device):
    model.eval()
    probs = []
    bar = tqdm(loader, desc="Predicting", leave=False)
    with torch.no_grad():
        for batch in bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            prob = F.softmax(outputs.logits, dim=1)[:, 1]
            probs.extend(prob.cpu().numpy())
    return probs

# save
pred_probs = predict_proba(model, test_loader, device)
ids = test_df['ID'].tolist() if 'ID' in test_df.columns else list(range(len(test_df)))
sub_df = pd.DataFrame({'ID': ids, 'generated': pred_probs})
sub_path = f"submission.csv"
sub_df.to_csv(sub_path, index=False)
print(f"✅ {sub_path} 저장 완료")
