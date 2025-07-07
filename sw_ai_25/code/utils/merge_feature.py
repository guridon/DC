import pickle
import pandas as pd
import numpy as np
import argparse
import os
from tqdm import tqdm
import re

def merge_feature(prefix, input_dir, start, end, output_dir, file_name):
    data_path = f"../../data/{prefix}_{input_dir}"
    start=f"{prefix}_{start}"
    pkl_files = [fname for fname in os.listdir(data_path) if fname.startswith(start) and fname.endswith(end)]
    def natural_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]
    pkl_files.sort(key=natural_key)
    print(pkl_files)

    if not pkl_files:
        print(f"파일 ❌")
        return

    dfs = []
    for fname in tqdm(pkl_files, desc="paragraph_pos_idx.pkl 병합: "):
        df = pd.read_pickle(os.path.join(data_path, fname))
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)

    drop_list = ["title", "paragraph_index", "paragraph_pos"]
    all_df.drop(drop_list, axis=1, inplace=True)
    print(f"data points: {len(all_df)} 개")

    feature_list = ["NNG", "VV", "JKS", "JKO", "EF", "EC", "NNP", "SL",
                    "josa_variety", "eomi_variety", "josa_repeat",
                    "eomi_repeat", "pos_type_variety"]
    
    
    print(f"len(feature_list): {len(feature_list)}")
    # feature_matrix = all_df[feature_list].to_numpy()
    # print(f"feature_matrix.shape: {feature_matrix.shape}")

    output_path = f"../../data/{prefix}_{output_dir}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_full_name = os.path.join(output_path, file_name)
    # if not output_full_name.endswith('.npy'):
    #     output_full_name += '.npy'
    # np.save(output_full_name, feature_matrix)

    all_df.to_pickle(output_full_name)
    print(f"[Saved] {output_full_name} 저장 완료 🗂️")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="형 변환")
    parser.add_argument('--prefix', type=str, default="train")
    parser.add_argument('--input_dir', type=str, default="paragraph_pos_stats")
    parser.add_argument('--start', type=str, default="paragraph_pos_")
    parser.add_argument('--end', type=str, default=".pkl")
    parser.add_argument('--output_dir', type=str, default='feature_df')
    parser.add_argument('--file_name', type=str, default="paragraph_pos_stats.pkl")
    args = parser.parse_args()

    merge_feature(prefix=args.prefix,
              input_dir=args.input_dir,
              start=args.start,
              end=args.end,
              output_dir=args.output_dir,
              file_name=args.file_name)
