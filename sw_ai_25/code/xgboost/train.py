import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from dataset import Dataset
from feature import ConcatFeatureDF
from utils import pick_list

from xgboost import XGBClassifier

import argparse

def main():
    dataset = Dataset()
    dataset.load_pickle()
    dataset.set_list()
    feature_df_name_list = ["paragraph_pos_stats.pkl"]
    dataset.concat_feature_df(feature_df_name_list)
    dataset.split_data()
    dataset.make_matrix()
    train_full_matrix, val_full_matrix = dataset.get_train_matrix()
    test_full_matrix =  dataset.get_test_matrix()

    xgb = XGBClassifier(random_state=42)
    print(f"====================================================")
    print(f"[Train] training...")
    xgb.fit(train_full_matrix, dataset.y_train)

    val_probs = xgb.predict_proba(val_full_matrix)[:, 1]
    auc = roc_auc_score(dataset.y_val, val_probs)
    print(f"Validation AUC: ⚠️ {auc:.4f} ⚠️")

    print(f"[Predict] ...")
    probs = xgb.predict_proba(test_full_matrix)[:, 1]

    print(f"[Inference] Making submission file ... 📝")
    sample_submission = pd.read_csv('../data/sample_submission.csv', encoding='utf-8-sig')
    sample_submission['generated'] = probs

    sample_submission.to_csv(f'../output/baseline_submission_para_change_ngram_sim_scaled.csv', index=False)
    print("---------------------------------------- DONE.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument('--dir_name', type=str, default="sentence_pkl")

    args=main()