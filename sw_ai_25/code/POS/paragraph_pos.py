import pandas as pd
import pickle
# from konlpy.tag import Mecab
#from mecab import MeCab

from kiwipiepy import Kiwi

from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing
import os

input_csv = "../../data/unprocessed_paragraph.csv"
output_dir = "../../data/paragraph_pos"
os.makedirs(output_dir, exist_ok=True)

batch_size = 150000  
batch_counter = 7
current_batch = []
current_count = 0
chunksize = 50000

def process_title_group(group):
    try:
        local_kiwi = Kiwi()
        new_group = group[['title', 'paragraph_index', 'generated']].copy()
        new_group['paragraph_pos'] = group['paragraph_text'].apply(
            lambda text: [(token.form, token.tag) for token in local_kiwi.analyze(text)[0][0]]
        )
        return new_group
    except Exception as e:
        print(f"[Error] title: {group['title'].iloc[0]} - {e}")
        return pd.DataFrame() 

try:
    with open(input_csv, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f) - 1
    total_chunks = (total_lines // chunksize) + 1

    reader = pd.read_csv(input_csv, chunksize=chunksize)
    with tqdm(total=total_chunks, desc="Chunk processing") as pbar:
        for chunk in reader:
            title_groups = [group for _, group in chunk.groupby('title')]
            group_sizes = [len(g) for g in title_groups]
            idx = 0
            while idx < len(title_groups):
                group = title_groups[idx]
                group_size = group_sizes[idx]
                if current_count + group_size > batch_size:
                    print(f"[Batch {batch_counter}] Processing {len(current_batch)} title groups...")
                    processed = Parallel(n_jobs=multiprocessing.cpu_count())(
                        delayed(process_title_group)(g) for g in tqdm(current_batch, desc="Title group POS", leave=False)
                    )
                    batch_df = pd.concat(processed, ignore_index=True)
                    batch_df.to_pickle(os.path.join(output_dir, f"paragraph_pos_{batch_counter}.pkl"))
                    print(f"[Batch {batch_counter}] Saved {len(batch_df)} rows.")
                    batch_counter += 1
                    current_batch = [group]
                    current_count = group_size
                else:
                    current_batch.append(group)
                    current_count += group_size
                idx += 1
            pbar.update(1)

    if current_batch:
        print(f"[Batch {batch_counter}] Processing {len(current_batch)} title groups (last batch)...")
        processed = Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(process_title_group)(g) for g in tqdm(current_batch, desc="Title group POS", leave=False)
        )
        batch_df = pd.concat(processed, ignore_index=True)
        batch_df.to_pickle(os.path.join(output_dir, f"paragraph_pos_{batch_counter}.pkl"))
        print(f"[Batch {batch_counter}] Saved {len(batch_df)} rows.")

except Exception as e:
    print(f"[Fatal Error] {e}")
