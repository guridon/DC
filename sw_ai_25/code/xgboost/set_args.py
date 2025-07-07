import argparse

def set_args():
    parser = argparse.ArgumentParser(description="train")

    parser.add_argument('--train_file_name', type=str, default="../../data/train_paraemb_change_ngram.pkl")
    parser.add_argument('--test_file_name', type=str, default="../../data/test_paraemb_change_ngram.pkl")
    parser.add_argument('--log_dir', type=str, default="./log")

    parser.add_argument('--train_feature_path', type=str, default="../../data/train_feature_df")
    parser.add_argument('--test_feature_path', type=str, default="../../data/test_feature_df")    

    args=parser.parse_args()
    return args