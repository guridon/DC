import os
import argparse
import pandas as pd

def merge_sentence_files(input_dir, idx):
    file_prefix = "sentence_"
    file_suffix = ".pkl"
    file_current = os.path.join(input_dir, f"{file_prefix}{idx}{file_suffix}")
    file_next = os.path.join(input_dir, f"{file_prefix}{idx+1}{file_suffix}")

    if not os.path.exists(file_current) or not os.path.exists(file_next):
        print(f"파일 ❌: {file_current} 또는 {file_next}")
        return
    
    df_current = pd.read_pickle(file_current)
    df_next = pd.read_pickle(file_next)

    last_title_current = df_current.iloc[-1]['title']
    first_title_next = df_next.iloc[0]['title']

    if last_title_current == first_title_next:
        same_title_rows = df_next[df_next['title'] == first_title_next]
        df_next_remaining = df_next[df_next['title'] != first_title_next]

        df_current = pd.concat([df_current, same_title_rows], ignore_index=True)

        df_current['sentence_index'] = df_current.groupby('title').cumcount()
        df_current.to_pickle(file_current)
        df_next_remaining.to_pickle(file_next)
        print(f"[{idx}, {idx+1}] '{first_title_next}' 병합: {len(same_title_rows)}개 행 이동")
    else:
        print(f"[{idx}, {idx+1}] 병합 ❌")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sentence_{idx}.pkl 파일 병합")
    parser.add_argument('--idx', type=int, required=True)
    parser.add_argument('--input_dir', type=str, default='../../data/sentence_pkl')
    args = parser.parse_args()

    merge_sentence_files(args.input_dir, args.idx)
