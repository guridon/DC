import argparse

def set_args():
    parser = argparse.ArgumentParser(description="train")

    parser.add_argument('--train_file_path', type=str, default="../../data/train_synthetic")
    parser.add_argument('--train_file_name', type=str, default="paragraph_1.pkl")
    parser.add_argument('--test_file_path', type=str, default="../../data/test")
    parser.add_argument('--test_file_name', type=str, default="test_base.pkl")

    parser.add_argument('--log_dir', type=str, default="./log")

    parser.add_argument('--train_feature_path', type=str, default="../../data/train_feature_df")
    parser.add_argument('--test_feature_path', type=str, default="../../data/test_feature_df") 

    parser.add_argument('--output_path', type=str, default="./output") 

    parser.add_argument('--tfidf', action='store_true', help="TF-IDF 사용 여부") 

    args=parser.parse_args()
    return args