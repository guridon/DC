from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import numpy as np

class FeatureEngineering:
    def __init__(self, train_matrix, val_matrix, test_matrix):
        self.train_matrix = train_matrix
        self.val_matrix = val_matrix
        self.test_matrix = test_matrix
        self.feature_df = None
        
    
    def scaled_matrix(self, matrix):
        scaler = StandardScaler()
        matrix_scaled = scaler.fit_transform(matrix)
        return matrix_scaled

    # TODO
    """
    def feature enginnering() -> Dataframe:
        특징 추출..
        
    def make_feature_df() -> Dataframe:
        ..
        return feature_df 

    etc..
    """


    def concat_feature(self, matrix, feature_df):
        if feature_df is None:
            feature_df = self.feature_df
        col_list=list(feature_df.columns)
        feature_matrix=feature_df[col_list].to_numpy()
        # feature_matrix_scared = self.scaled_matrix(feature_matrix)
        return np.hstack([matrix, feature_matrix])
        

class ConcatFeatureDF:
    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df
        self.feature_list=[]
        self.feature_info = {}

    def pick_and_concat(self, feature_df_name_list, mode='train'):
        """
        feature_df_name_list: feature DataFrame 파일명 리스트
        mode: 'train' 또는 'test'
        """
        data_path = f"../../data/{mode}_feature_df"
        base_df = self.train_df if mode == 'train' else self.test_df

        for feature_df_name in feature_df_name_list:
            feature_df = pd.read_pickle(os.path.join(data_path, feature_df_name))
            print(f"\n[{feature_df_name}]")
            for idx, col in enumerate(feature_df.columns):
                print(f"{idx}: {col}")
            selected = input("사용할 feature의 번호를 콤마(,)로 구분해서 입력 (ex: 0,2,3, 또는 all): ").strip()
            if selected.lower() == "all":
                chosen_df = feature_df
                chosen_cols = list(feature_df.columns)
            else:
                selected_idx = [int(x.strip()) for x in selected.split(",") if x.strip().isdigit()]
                chosen_cols = [feature_df.columns[i] for i in selected_idx]
                chosen_df = feature_df[chosen_cols]
            base_df = pd.concat([base_df, chosen_df], axis=1)
            self.feature_list.extend(chosen_cols)
            self.feature_info[feature_df_name] = chosen_cols

        if mode == 'train':
            self.train_df = base_df
        else:
            self.test_df = base_df
        
        return self.feature_list

    def get_train(self):
        return self.train_df

    def get_test(self):
        return self.test_df
    
    def log_feature_info_txt(self, log_path="./log/feature.txt"):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'a', encoding='utf-8') as f:
            for fname, cols in self.feature_info.items():
                f.write(f"{fname}: {', '.join(cols)}\n")


