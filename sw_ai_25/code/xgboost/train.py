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
    dataset = Dataset("../../data/train_paraemb_change_ngram.pkl", "../../data/test_paraemb_change_ngram.pkl")
    train, test = dataset.load_train_pickle()
    train_emb_list, test_emb_list, feature_list = pick_list(list(train.columns))
    dataset.set_list(train_emb_list, test_emb_list, feature_list)
    ########################### concat feature_df ##############################
    feature_df_name_list=["paragraph_pos_stats.pkl"]
    concat_feature_df = ConcatFeatureDF(train, test)
    c_feature_list = concat_feature_df.pick_and_concat(feature_df_name_list, mode="train")
    concat_feature_df.pick_and_concat(feature_df_name_list, mode="test")
    train, test = concat_feature_df.get_train(), concat_feature_df.get_test()
    ############################################################################
    X_train, X_val, y_train, y_val = dataset.split_data(train)

    dataset.df_load_train_text_emb(X_train, X_val, train_emb_list)
    train_full_matrix, val_full_matrix= dataset.concat_train_feature(c_feature_list)
    
    xgb = XGBClassifier(random_state=42)
    xgb.fit(train_full_matrix, y_train)

    val_probs = xgb.predict_proba(val_full_matrix)[:, 1]
    auc = roc_auc_score(y_val, val_probs)
    print(f"Validation AUC: {auc:.4f}")

    dataset.df_load_test_text_emb(test, test_emb_list)
    test_full_matrix = dataset.concat_test_feature()
    probs = xgb.predict_proba(test_full_matrix)[:, 1]
    sample_submission = pd.read_csv('../data/sample_submission.csv', encoding='utf-8-sig')
    sample_submission['generated'] = probs

    sample_submission.to_csv(f'../output/baseline_submission_para_change_ngram_sim_scaled.csv', index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument('--dir_name', type=str, default="sentence_pkl")

    args=main()