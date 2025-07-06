import os
import pandas as pd
import argparse
import re
import numpy as np

def natural_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

def merge_pickle_files(prefix, dir_name, start_str, end_str, output_dir):
    dir_path="../../data"
    dir_fullname=f"{prefix}_{dir_name}"
    
    directory=os.path.join(dir_path,dir_fullname)

    files = os.listdir(directory)
    filtered_files = [f for f in files if f.startswith(start_str) and f.endswith(end_str)]
    filtered_files.sort(key=natural_key)
    
    merged_df = pd.DataFrame()
    for file in filtered_files:
        file_path = os.path.join(directory, file)
        df = pd.read_pickle(file_path)
        merged_df = pd.concat([merged_df, df], ignore_index=True)
    
    merged_df.drop("sentence_text", axis=1, inplace=True)
    merged_df.insert(0,"SID",range(len(merged_df)))
    # merged_df['sentence_index'] = merged_df.groupby('title').cumcount()

    # 자료형 변환
    merged_df=merged_df.astype({"paragraph_index":np.int16,
                                "sentence_index":np.int16,
                                "generated":np.int16,
                                "sentence_emb":"float32"})

    # ../../data + out_dir(prefix) + out_file()
    output_fullname=f"{prefix}_{output_dir}/feature{end_str}"
    output_file=os.path.join(dir_path,output_fullname)
    merged_df.to_pickle(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sentence_{idx}.pkl 파일 병합")
    parser.add_argument('--prefix', type=str, default="train")
    parser.add_argument('--dir_name', type=str, default="sentence_pkl")
    parser.add_argument('--start_str', type=str, default="sentence_")
    parser.add_argument('--end_str', type=str, default=".pkl")
    parser.add_argument('--output_dir', type=str, default='feature')
    args = parser.parse_args()

    merged = merge_pickle_files(
                prefix=args.prefix,
                dir_name=args.dir_name,
                start_str=args.start_str,
                end_str=args.end_str,
                output_dir=args.output_dir
            )

