from dataset import Dataset
from set_args import set_args
from config import xgb_params
from utils import metrics

import sys
import pandas as pd
import os

from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import xgboost as xgb

import argparse

def main(args):
    dataset = Dataset(args)
    dataset.load_file()
    dataset.set_list()
    dataset.split_data()
    dataset.make_matrix()
    train_full_matrix, val_full_matrix = dataset.get_train_matrix()
    test_full_matrix =  dataset.get_test_matrix()
    dataset.log({
                "xgb_params": xgb_params,
                "python_version": sys.version.replace('\n', ' ')
            }| vars(args))

    xgb = XGBClassifier(**xgb_params)
    print(f"====================================================")
    print(f"[Train] training...")
    xgb.fit(train_full_matrix, dataset.y_train)

    val_probs = xgb.predict_proba(val_full_matrix)[:, 1]
    metrics(dataset.y_val, val_probs, dataset)

    print(f"[Predict] ...")
    probs = xgb.predict_proba(test_full_matrix)[:, 1]

    print(f"[Inference] Making submission file ...")
    sample_submission = pd.read_csv('../data/sample_submission.csv', encoding='utf-8-sig')
    sample_submission['generated'] = probs

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
    output_file=os.path.join(args.output_path, f"baseline_submission_{dataset.now}.csv")
    sample_submission.to_csv(output_file, index=False)
    print(f"✅[Saved] {output_file}")
    dataset.log(vars(args))

if __name__ == "__main__":
    args=set_args()
    main(args)