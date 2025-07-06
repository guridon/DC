import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import sent_tokenize

input_csv = "../../data/train_paragraph.csv"
print(f"[Load]{input_csv} 로드 완료 🗂️")
output_dir = "../../data/train_sentence_pkl_2"
os.makedirs(output_dir, exist_ok=True)

batch_size = 200000  
batch_counter = 1
acc_sentences = []
chunksize = 50000

MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def get_cls_embedding_batch(texts, tokenizer, model, device, max_length=256, batch_size=32):
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches", leave=False):
            batch_texts = texts[i:i+batch_size]
            if isinstance(batch_texts, np.ndarray):
                batch_texts = batch_texts.tolist()
            inputs = tokenizer(
                batch_texts,
                return_tensors='pt',
                truncation=True,
                max_length=max_length,
                padding='max_length'
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_emb)
    embeddings = np.vstack(embeddings)
    return embeddings

def paragraph_to_sentences(row):
    try:
        sentences = sent_tokenize(row['paragraph_text'])
        return [
            {
                'title': row['title'],
                'paragraph_index': row['paragraph_index'],
                'sentence_index': idx,
                'sentence_text': sent,
                'generated': row['generated']
            }
            for idx, sent in enumerate(sentences)
        ]
    except Exception as e:
        print(f"[Error] title: {row.get('title', 'N/A')} - {e}")
        return []

def process_title_group(group):
    results = []
    for _, row in group.iterrows():
        results.extend(paragraph_to_sentences(row))
    return results

try:
    with open(input_csv, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f) - 1
    total_chunks = (total_lines // chunksize) + 1

    reader = pd.read_csv(input_csv, chunksize=chunksize)
    with tqdm(total=total_chunks, desc="Chunk processing") as pbar:
        for chunk in reader:
            title_groups = [group for _, group in chunk.groupby('title', sort=False)]
            processed = Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(process_title_group)(g) for g in tqdm(title_groups, desc="Title group sentence split", leave=False)
            )
            for group_sentences in processed:
                acc_sentences.extend(group_sentences)
                # batch_size마다 저장
                while len(acc_sentences) >= batch_size:
                    df = pd.DataFrame(acc_sentences[:batch_size])
                    # 임베딩 
                    print(f"[Batch {batch_counter}] Extracting embeddings...")
                    try:
                        df['sentence_emb'] = list(get_cls_embedding_batch(
                            df['sentence_text'].tolist(), tokenizer, model, device
                        ))
                        df.to_pickle(os.path.join(output_dir, f"sentence_{batch_counter}.pkl"))
                        print(f"[Batch {batch_counter}] Saved {batch_size} sentences (with embeddings).")
                    except Exception as e:
                        print(f"[Batch {batch_counter}] Error during embedding or saving: {e}")
                    batch_counter += 1
                    acc_sentences = acc_sentences[batch_size:]
            pbar.update(1)
    # 마지막 문장 
    if acc_sentences:
        df = pd.DataFrame(acc_sentences)
        print(f"[Batch {batch_counter}] Extracting embeddings (last batch)...")
        try:
            df['sentence_emb'] = list(get_cls_embedding_batch(
                df['sentence_text'].tolist(), tokenizer, model, device
            ))
            df.to_pickle(os.path.join(output_dir, f"sentence_{batch_counter}.pkl"))
            print(f"[Batch {batch_counter}] Saved {len(df)} sentences (with embeddings).")
        except Exception as e:
            print(f"[Batch {batch_counter}] Error during embedding or saving (last batch): {e}")

except Exception as e:
    print(f"[Fatal Error] {e}")
