import pandas as pd
import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

class Dataset:
    def __init__(self, train_file_name, test_file_name):
        self.train_file_name = train_file_name
        self.test_file_name = test_file_name
        self.train_emb_list, self.test_emb_list = [], []
        self.text_list, self.feature_list = None, None
        self.train, self.test = None, None
        self.X_train, self.X_val = None, None
        self.y_train, self.y_val = None, None
        

        self.train_text_matrix, self.val_text_matrix = None, None
        self.test_text_matrix = None

        print(f"[Init] Dataset initialized with train: {train_file_name}, test: {test_file_name}")

    def get_cls_embedding_batch(self, texts, max_length=256, batch_size=32):
        print(f"[Embedding] Start embedding {len(texts)} texts (batch size: {batch_size})")
        MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="[Embedding batches]"):
                batch_texts = texts[i:i+batch_size]
                inputs = tokenizer(
                    batch_texts,
                    return_tensors='pt',
                    truncation=True,
                    max_length=max_length,
                    padding='max_length'
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_emb)
        embeddings = np.vstack(embeddings) 
        print(f"[Embedding] Completed. Shape: {embeddings.shape}")
        return embeddings
    
    def load_train_csv(self):
        print(f"[Load] Loading train CSV: {self.train_file_name}")
        train = pd.read_csv(self.train_file_name)
        print(f"[Load] Loading test CSV: {self.test_file_name}")
        test = pd.read_csv(self.test_file_name)
        print(f"[Load] Loaded train shape: {train.shape}, test shape: {test.shape}")
        return train, test

    def load_train_pickle(self):
        print(f"[Load] Loading train pickle: {self.train_file_name}")
        with open(self.train_file_name, "rb") as f:
            train = pickle.load(f)
        print(f"[Load] Loading test pickle: {self.test_file_name}")
        with open(self.test_file_name, "rb") as f:
            test = pickle.load(f)
        self.train, self.test = train, test
        print(f"[Load] train_col_list: {list(train.columns)}")
        print(f"[Load] test_col_list: {list(test.columns)}")
        return train, test

    def set_list(self, train_emb_list, test_emb_list, feature_list, text_list=[]):
        self.train_emb_list = train_emb_list
        self.test_emb_list = test_emb_list
        self.feature_list = feature_list
        self.text_list = text_list
        print(f"[Set] train_emb_list: {train_emb_list}, \n test_emb_list: {test_emb_list}, \nfeature_list: {feature_list}, \ntext_list: {text_list}")

    def split_data(self, train):
        print(f"[Split] Splitting data with stratify on 'generated'")
        print(f"[Split] train_emb_list: {self.train_emb_list}")
        print(f"[Split] feature_list: {self.feature_list}")
        print(f"[Split] text_list: {self.text_list}")

        col_list = self.train_emb_list + self.feature_list + self.text_list
        X = train[col_list]
        y = train['generated']
        X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        print(f"[Split] X_train: {X_train.shape}, X_val: {X_val.shape}")
        self.X_train, self.X_val = X_train, X_val
        self.y_train, self.y_val = y_train, y_val
        
        return X_train, X_val, y_train, y_val
    
    def get_text_emb(self, text_list):
        if text_list is None:
            text_list = self.text_list
        text_set = []
        print(f"[Text Embedding] Extracting embeddings for: {text_list}")
        for text in text_list:
            print(f"[Text Embedding] Embedding column: {text}")
            tmp = self.get_cls_embedding_batch(self.train[text].tolist())
            text_set.append(tmp)
        print(f"[Text Embedding] All embeddings extracted. Shape: {[arr.shape for arr in text_set]}")
        return np.hstack(text_set)

    def df_load_train_text_emb(self, X_train, X_val, train_emb_list):
        if train_emb_list is None:
            train_emb_list = self.train_emb_list
        print(f"[DF Load] Loading train/val text embeddings for: {train_emb_list}")
        train_matrix_set = []
        val_matrix_set = []
        for col in train_emb_list:
            print(f"[DF Load] Processing train column: {col}")
            tmp = np.vstack(X_train[col].tolist())
            train_matrix_set.append(tmp)
        for col in train_emb_list:
            print(f"[DF Load] Processing val column: {col}")
            tmp = np.vstack(X_val[col].tolist())
            val_matrix_set.append(tmp)
        train_text_matrix = np.hstack(train_matrix_set)
        val_text_matrix = np.hstack(val_matrix_set)
        self.train_text_matrix, self.val_text_matrix = train_text_matrix, val_text_matrix
        print(f"[DF Load] train_text_matrix shape: {train_text_matrix.shape}, val_text_matrix shape: {val_text_matrix.shape}")
        # return train_text_matrix, val_text_matrix
    
    def df_load_test_text_emb(self, X_test, test_emb_list):
        if test_emb_list is None:
            test_emb_list = self.test_emb_list
        print(f"[DF Load Test] Loading test text embeddings for: {test_emb_list}")
        test_matrix_set = []
        for col in test_emb_list:
            print(f"[DF Load Test] Processing test column: {col}")
            tmp = np.vstack(X_test[col].tolist())
            test_matrix_set.append(tmp)
        test_text_matrix = np.hstack(test_matrix_set)
        self.test_text_matrix=test_text_matrix
        print(f"[DF Load Test] test_text_matrix shape: {self.test_text_matrix.shape}")
        # return test_text_matrix
    
    def scaled_matrix(self, matrix):
        scaler = StandardScaler()
        matrix_scaled = scaler.fit_transform(matrix)
        return matrix_scaled

    def concat_train_feature(self, c_feature_list=[]):
        self.feature_list = list(dict.fromkeys(self.feature_list + c_feature_list))   # 중복 처리
        print(f"[Concat] Concatenating features: {self.feature_list}")
        train_feature_matrix = self.X_train[self.feature_list].to_numpy()
        val_feature_matrix = self.X_val[self.feature_list].to_numpy()

        # train_feature_matrix_scaled = self.scaled_matrix(train_feature_matrix)
        # val_feature_matrix_scaled = self.val_matrix(train_feature_matrix)

        train_full_matrix = np.hstack([self.train_text_matrix, train_feature_matrix])
        val_full_matrix = np.hstack([self.val_text_matrix, val_feature_matrix])
        print(f"[Concat] train_full_matrix shape: {train_full_matrix.shape}, val_full_matrix shape: {val_full_matrix.shape}")
        return train_full_matrix, val_full_matrix

    def concat_test_feature(self):
        print(f"[Concat Test] Concatenating test features: {self.feature_list}")
        test_feature_matrix = self.test[self.feature_list].to_numpy()
        # test_feature_matrix_scaled = self.scaled_matrix(test_feature_matrix)
        test_full_matrix = np.hstack([self.test_text_matrix, test_feature_matrix])
        print(f"[Concat Test] test_full_matrix shape: {test_full_matrix.shape}")
        return test_full_matrix