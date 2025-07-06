import os
import argparse
import pandas as pd

def make_full_name(input_dir, output_dir, idx):
    file_prefix = "sentence_"
    file_suffix = ".pkl"
    input_file_current = os.path.join(input_dir, f"{file_prefix}{idx}{file_suffix}")
    input_file_next = os.path.join(input_dir, f"{file_prefix}{idx+1}{file_suffix}")

    output_file_current = os.path.join(output_dir, f"{file_prefix}{idx}{file_suffix}")
    output_file_next = os.path.join(output_dir, f"{file_prefix}{idx+1}{file_suffix}")

    return input_file_current, input_file_next, output_file_current, output_file_next

def merge_sentence_files(input_dir, output_dir, idx):
    os.makedirs(output_dir, exist_ok=True)
    input_file_current, input_file_next, output_file_current, output_file_next = make_full_name(input_dir, output_dir, idx)

    if not os.path.exists(input_file_current) or not os.path.exists(input_file_next):
        print(f"파일 ❌: {input_file_current} 또는 {input_file_next}")
        return
    
    df_current = pd.read_pickle(input_file_current)
    df_next = pd.read_pickle(input_file_next)

    last_title_current = df_current.iloc[-1]['title']
    first_title_next = df_next.iloc[0]['title']

    if last_title_current == first_title_next:
        len_current_before = len(df_current)
        len_next_before = len(df_next)

        same_title_rows = df_next[df_next['title'] == first_title_next]
        df_next_remaining = df_next[df_next['title'] != first_title_next]

        df_current = pd.concat([df_current, same_title_rows], ignore_index=True)

        len_current_after = len(df_current)
        len_next_after = len(df_next_remaining)

        df_current['sentence_index'] = df_current.groupby('title').cumcount()
        df_current.to_pickle(output_file_current)
        df_next_remaining.to_pickle(output_file_next)

        print(
                f"[{idx}, {idx+1}] '{first_title_next}' 병합: {len(same_title_rows)}개 행 이동\n"
                f"  - df_current: {len_current_before} → {len_current_after}\n"
                f"  - df_next: {len_next_before} → {len_next_after}"
            )
    else:
        print(f"[{idx}, {idx+1}] 병합 ❌")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sentence_{idx}.pkl 파일 병합")
    parser.add_argument('--idx', type=int, required=True)
    parser.add_argument('--input_dir', type=str, default='../../data/train_sentence_pkl_original')
    parser.add_argument('--output_dir', type=str, default='../../data/train_sentence_chuck_pkl')

    args = parser.parse_args()

    merge_sentence_files(args.input_dir, args.output_dir, args.idx)
