# env: zeroshot
import os
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch

from pytod.models.lof import LOF as GPU_LOF
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_sentence_files(input_dir, idx):
    file_prefix = "sentence_"
    file_suffix = ".pkl"
    file_current = os.path.join(input_dir, f"{file_prefix}{idx}{file_suffix}")

    if not os.path.exists(file_current):
        print(f"파일 ❌: {file_current}")
        return

    # col: title, paragraph_index, sentence_index, sentence_text, generated, sentence_emb
    df = pd.read_pickle(file_current)
    df['lof'] = 0.0  # 미리 컬럼 생성

    features = []
    total_titles = df['title'].nunique()
    for title, group in tqdm(df.groupby('title', sort=False), desc='Processing title...',
                             total=total_titles):
        try:
            emb_matrix = np.vstack(group['sentence_emb'].values)  # (문장수, 임베딩차원)
            if emb_matrix.shape[0] < 2:
                features.append([title, 0, 0, 0, 0])
                df.loc[group.index, 'lof'] = 0.0
            else:
                emb_tensor = torch.tensor(emb_matrix, dtype=torch.float32).to(device)  # GPU로 이동
                lof = GPU_LOF(n_neighbors=min(3, emb_tensor.shape[0]-1))
                lof.fit(emb_tensor)
                lof_scores = lof.decision_scores_  
                lof_scores = np.nan_to_num(lof_scores, nan=0.0, posinf=0.0, neginf=0.0)
                if hasattr(lof_scores, "cpu"):
                    lof_scores = lof_scores.cpu().numpy()
                lof_pred = lof.labels_
                if hasattr(lof_pred, "cpu"):
                    lof_pred = lof_pred.cpu().numpy()
                df.loc[group.index, 'lof'] = lof_scores.astype('float32')

                max_lof = np.max(lof_scores)
                mean_lof = np.mean(lof_scores)
                std_lof = np.std(lof_scores)
                outlier_ratio = np.mean(lof_pred)
                features.append([title, max_lof, mean_lof, std_lof, outlier_ratio])

        except Exception as e:
            print(f'Error processing title {title}: {e}')
            features.append([title, 0, 0, 0, 0])
            df.loc[group.index, 'lof'] = 0.0

    title_lof_df = pd.DataFrame(features, 
                columns=['title', 'max_lof','mean_lof', 'std_lof',
                'outlier_ratio']).astype({'max_lof': 'float32', 
                                            'mean_lof': 'float32', 
                                            'std_lof': 'float32', 
                                        'outlier_ratio': 'float32'})
    # save
    os.makedirs("../../feature/", exist_ok=True)
    df.to_pickle(file_current)
    title_lof_df.to_pickle("../../feature/title_lof_df.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sentence_{idx}.pkl 파일 병합")
    parser.add_argument('--idx', type=int, required=True)
    parser.add_argument('--input_dir', type=str, default='../../data/sentence_pkl')
    args = parser.parse_args()

    load_sentence_files(args.input_dir, args.idx)
