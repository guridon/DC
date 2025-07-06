import pickle
import pandas as pd
import numpy as np
import argparse

import os
import re

def df_astype(prefix, input_dir, idx, output_dir):
    data_path=f"../../data/{prefix}_{input_dir}"
    file_name=f"{prefix}_paragraph_pos_{idx}.pkl"
    input_full_name=os.path.join(data_path, file_name)
    
    if not os.path.exists(input_full_name):
        print(f"파일 ❌: {input_full_name}")
        return
    
    df=pd.read_pickle(input_full_name)
    print(f"{file_name} data points: {len(df)} 개")

    df = df.astype({"PID":"int32", 
                    "paragraph_index":np.int16, 
                    "generated": np.int16})

    output_path=f"../../data/{prefix}_{output_dir}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_full_name=os.path.join(output_path, file_name)
    df.to_pickle(output_full_name)
    print(f"[Saved] {output_full_name} 저장 완료 🗂️")
    


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="형 변환")
    parser.add_argument('--prefix', type=str, default="train")
    parser.add_argument('--input_dir', type=str, default="pid_paragraph_pos")
    parser.add_argument('--idx', type=int, required=True)
    parser.add_argument('--output_dir', type=str, default='paragraph_pos')
    args = parser.parse_args()

    df_astype(prefix=args.prefix, 
              input_dir=args.input_dir, 
              idx=args.idx, output_dir=args.output_dir)
