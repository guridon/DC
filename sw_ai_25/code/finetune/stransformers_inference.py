import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from tqdm import tqdm
import safetensors.torch

class KoELECTRAMeanPoolingClassifier(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (batch, seq, hidden)
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        masked = last_hidden * mask
        summed = torch.sum(masked, 1)
        summed_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / summed_mask
        logits = self.classifier(mean_pooled)
        return logits

class TestParagraphDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.data = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data.loc[idx]
        paragraph_text = item["paragraph_text"]
        encoding = self.tokenizer(
            paragraph_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )
        out = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }
        return out

# config
num=4126
# checkpoint_dir = f"./checkpoint_koelectra_meanpooling_trainer_qwen_2500paired/checkpoint-{num}"
checkpoint_dir = "checkpoint_koelectra_meanpooling_trainer_aug_batch1_best"
max_length = 512
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    device_str = f"cuda:{torch.cuda.current_device()}"
else:
    device_str = "cpu"

tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
model = KoELECTRAMeanPoolingClassifier("monologg/koelectra-base-v3-discriminator", num_labels=2)
state_dict = safetensors.torch.load_file(
    os.path.join(checkpoint_dir, "model.safetensors"),
    device=device_str
)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# test dataset & dataloader
test_df = pd.read_csv('../../data/test.csv')
test_dataset = TestParagraphDataset(test_df, tokenizer, max_length=max_length)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def predict_proba(model, loader, device):
    probs = []
    bar = tqdm(loader, desc="Predicting", leave=False)
    with torch.no_grad():
        for batch in bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            prob = F.softmax(logits, dim=1)[:, 1]
            probs.extend(prob.cpu().numpy())
    return probs

# inference && save
pred_probs = predict_proba(model, test_loader, device)
ids = test_df['ID'].tolist() if 'ID' in test_df.columns else list(range(len(test_df)))
sub_df = pd.DataFrame({'ID': ids, 'generated': pred_probs})
sub_path = f"submission_baseline_finetune_aug_batch1.csv"
sub_df.to_csv(sub_path, index=False)
print(f"✅ {sub_path} 저장 완료")
