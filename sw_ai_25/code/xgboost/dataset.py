import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import pickle
from datetime import datetime

from transformers import AutoTokenizer, AutoModel
import torch
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

class Dataset:
    def __init__(self, args):
        self.args=args
        self.train_file_name, self.test_file_name = self.get_file_name()
        self.train_feature_path = self.args.train_feature_path
        self.test_feature_path = self.args.test_feature_path

        self.train_emb_list, self.test_emb_list = [], []
        self.text_list, self.feature_list = [], []
        self.train, self.test = None, None
        self.X_train, self.X_val = None, None
        self.y_train, self.y_val = None, None

        self.tfidf = self.args.tfidf
        self.vectorizer = None
        self.train_text_matrix, self.val_text_matrix, self.test_text_matrix = None, None, None
        self.train_full_matrix, self.val_full_matrix, self.test_full_matrix = None, None, None
        
        self.feature_info = {}
        self.log_path, self.now = self.set_log_path(self.args.log_dir)
        self.log(f"실험시각: {self.now}")

        print(f"[Init] Dataset initialized with train: {self.train_file_name}, test: {self.test_file_name}")

    def set_log_path(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"feature_{now}.txt")
        return log_path, now

    def get_file_name(self):
        paths=[self.args.train_file_path, self.args.test_file_path]
        names= [self.args.train_file_name, self.args.test_file_name]
        files=[]
        for path, file_name in zip(paths, names):
            name, suffix = os.path.splitext(file_name)
            if suffix==".pkl":
                emb_file_name=f"{name}_{self.args.emb}{suffix}"
            else:
                emb_file_name=f"{name}{suffix}"
            full_file_name=os.path.join(path, emb_file_name)
            files.append(full_file_name)
        return files

    def get_meanpool_embedding_batch(self, texts, max_length=256, batch_size=32):
        print(f"[Embedding] Start embedding {len(texts)} texts (batch size: {batch_size})")
        MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        embeddings = []
        # model.eval()
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding (batch)"):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors='pt',
                truncation=True,
                max_length=max_length,
                padding='max_length'
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
                mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
                summed = (last_hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1)
                mean_pooled = (summed / counts).cpu().numpy()  # (batch, hidden_dim)
                embeddings.append(mean_pooled)
        return np.vstack(embeddings)
    
    def save_emb_text_from_csv(self):
        emb_df=self.train[self.text_list+["generated"]]
        save_emb_path = os.path.join(self.args.train_file_path, "paragraph_emb_mp.pkl")
        emb_df.to_pickle(save_emb_path)
        print(f"[Saved] 임베딩 paragraph_emb_mp.pkl 저장 완료")
        text_df=self.train[self.text_list]
        save_text_path = os.path.join(self.args.train_file_path, "paragraph_text.csv")
        text_df.to_csv(save_text_path)
    
    def load_train_csv(self):
        print(f"[Load csv 🗂️] Loading train: {self.train_file_name}")
        self.train = pd.read_csv(self.train_file_name)
        print(f"[Load] Loaded train shape: {self.train.shape}")
        print(f"[Pick: ✅ train text_list 선택]", end=" ")
        self.text_list = self.pick_columns("train.csv", self.train.columns)
        for col in self.text_list:
            self.train[f"{col}_emb_mp"] = list(self.get_meanpool_embedding_batch(self.train[col].tolist()))
        self.save_emb_text_from_csv()
        with open(self.test_file_name, "rb") as f:
            self.test = pickle.load(f)
        
    def load_train_pickle(self):
        print(f"[Load pkl 🥒] Loading train: {self.train_file_name}")
        with open(self.train_file_name, "rb") as f:
            self.train = pickle.load(f)
        print(f"[Load pkl 🥒] Loading test: {self.test_file_name}")
        with open(self.test_file_name, "rb") as f:
            self.test = pickle.load(f)
        if self.tfidf:
            print("[TF-IDF] Concat Text DF ")
            text_file = os.path.join(self.args.train_file_path, self.args.text_file_name)
            paragraph_text=pd.read_csv(text_file)
            print(f"[TF-IDF] Load text file {text_file}")
            self.text_list=self.pick_columns("paragraph_text", paragraph_text.columns)
            print(f" 🛠️ [Set] Text_list: {self.text_list}")
            self.train=pd.concat([self.train, paragraph_text[self.text_list]], axis=1)
            
    def load_file(self):
        suffix = os.path.splitext(self.train_file_name)[1]
        if suffix==".csv":
            self.load_train_csv()
        else:
            self.load_train_pickle()

    def pick_columns(self, feature_df_name, feature_df_cols):
        print(f"✨✨🌟{feature_df_name}🌟✨✨")
        for idx, col in enumerate(feature_df_cols):
            print(f"📌 {idx}: {col}")
        selected = input("💻 사용할 col 번호 (,)로 구분해서 입력 (ex: 0,2,3, 또는 all): ").strip()
        if selected == "":
            print("선택된 feature ❌")
            self.log(f"[{feature_df_name}] 선택된 feature ❌")
            return []
        if selected.lower() == "all":
            chosen_cols = list(feature_df_cols)
        else:
            selected_idx = [int(x.strip()) for x in selected.split(",") if x.strip().isdigit()]
            chosen_cols = [feature_df_cols[i] for i in selected_idx]
        self.log(f"[{feature_df_name}] 선택된 feature: {chosen_cols}")
        return chosen_cols

    def set_list(self):
        print(f"[Pick: ✅ train 임베딩 선택]", end=" ")
        train_emb_chosen_cols = self.pick_columns(self.train_file_name, self.train.columns)
        print(f"[Pick: ✅ test 임베딩 선택]", end=" ")
        test_emb_chosen_cols = self.pick_columns(self.test_file_name, self.test.columns)
        print(f"[Pick: ✅ feature]", end=" ")
        feat_chosen_cols = self.pick_columns(self.train_file_name, self.train.columns)
        
        self.train_emb_list = train_emb_chosen_cols
        self.test_emb_list = test_emb_chosen_cols
        self.feature_list = feat_chosen_cols
        print(f" 🛠️ [Set] 🟢 train_emb_list: {train_emb_chosen_cols}")
        print(f" 🛠️ [Set] 🔵 test_emb_list: {test_emb_chosen_cols}")
        print(f" 🛠️ [Set] 🩶 feature_list: {feat_chosen_cols}")


    def concat_feature_df(self, feature_df_name_list):
        base_train_df = self.train
        base_test_df = self.test

        for feature_df_name in feature_df_name_list:
            # train feature
            train_feature_df = pd.read_pickle(os.path.join(self.train_feature_path, feature_df_name))
            print("[Pick: ✅ feature_df feature 선택]")
            chosen_cols = self.pick_columns(feature_df_name, train_feature_df.columns) 
           
            chosen_train_df = train_feature_df[chosen_cols]
            base_train_df = pd.concat([base_train_df, chosen_train_df], axis=1)

            # test feature
            test_feature_df = pd.read_pickle(os.path.join(self.test_feature_path, feature_df_name))
            chosen_test_df = test_feature_df[chosen_cols]
            base_test_df = pd.concat([base_test_df, chosen_test_df], axis=1)

            self.feature_list.extend(chosen_cols)
            self.feature_info[feature_df_name] = chosen_cols

        self.train = base_train_df
        self.test = base_test_df

        print(f"self.train.columns: {self.train.columns}")
        print(f"self.test.columns: {self.test.columns}")

    def split_data(self):
        X = self.train
        y = self.train['generated']
        X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        print(f"⚔️💔[Split] X_train: {X_train.shape}, X_val: {X_val.shape}💔⚔️")
        self.X_train, self.X_val = X_train, X_val
        self.y_train, self.y_val = y_train, y_val

    def df_load_train_text_emb(self):
        train_matrix_set = []
        val_matrix_set = []
        for col in self.train_emb_list:
            tmp = np.vstack(self.X_train[col].tolist())
            train_matrix_set.append(tmp)
        for col in self.train_emb_list:
            tmp = np.vstack(self.X_val[col].tolist())
            val_matrix_set.append(tmp)
        self.train_text_matrix = np.hstack(train_matrix_set)
        self.val_text_matrix = np.hstack(val_matrix_set)
        print(f"🟢[Load: train emb] train_text_matrix shape: {self.train_text_matrix.shape}")
        print(f"🟢[Load: train emb] val_text_matrix shape: {self.val_text_matrix.shape}")

    def df_load_test_text_emb(self):
        test_matrix_set = []
        for col in self.test_emb_list:
            tmp = np.vstack(self.test[col].tolist())
            test_matrix_set.append(tmp)
        test_text_matrix = np.hstack(test_matrix_set)
        self.test_text_matrix=test_text_matrix
        print(f"🔵[Load: test emb] test_text_matrix shape: {self.test_text_matrix.shape}")

    def concat_tfidf(self):
        tfidf_train = self.X_train[self.text_list]
        tfidf_val = self.X_val[self.text_list]
        tfidf_test = self.test[self.text_list]
        import time
        print("🟣[TF-IDF] TF-IDF fitting ...")
        start = time.time()
        if len(self.text_list) == 1:
            col = self.text_list[0]
            tfidf_train=tfidf_train[col]
            tfidf_val=tfidf_val[col]
            tfidf_test=tfidf_test[col]
            if col=="title":
                max_features=3000
            else:
                max_features=10000
            self.vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=max_features)
        else:
            get_map={}
            for text_col in self.text_list:
                get_map[text_col] = FunctionTransformer(lambda x, col=text_col: x[col], validate=False)
            feature_union_list=[]
            for text_col, functiontransformer in get_map.items():
                if text_col=="title":
                    max_features = 3000
                else:
                    max_features = 10000
                feature_union_list.append((text_col, Pipeline([('selector', functiontransformer), 
                                            ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=max_features))])))
            self.vectorizer = FeatureUnion(feature_union_list)
        
        train_tfidf_matrix = self.vectorizer.fit_transform(tfidf_train)
        val_tfidf_matrix = self.vectorizer.transform(tfidf_val)
        test_tfidf_matrix = self.vectorizer.transform(tfidf_test)
        print(f"🟣[TF-IDF] TF-IDF fitting 완료; 소요 시간: {time.time() - start:.2f}초")

        # numpy2sparse
        train_full_sparse = sparse.csr_matrix(self.train_full_matrix)
        val_full_sparse = sparse.csr_matrix(self.val_full_matrix)
        test_full_sparse = sparse.csr_matrix(self.test_full_matrix)
        
        self.train_full_matrix = sparse.hstack([train_full_sparse, train_tfidf_matrix])
        self.val_full_matrix = sparse.hstack([val_full_sparse, val_tfidf_matrix])
        self.test_full_matrix = sparse.hstack([test_full_sparse, test_tfidf_matrix])

        print(f"🟢[Concat] train_tfidf_matrix shape: {self.train_full_matrix.shape}")
        print(f"🟢[Concat] val_tfidf_matrix shape: {self.val_full_matrix.shape}")
        print(f"🔵[Concat] test_tfidf_matrix shape: {self.test_full_matrix.shape}")
    
    def scaled_matrix(self, matrix):
        scaler = StandardScaler()
        matrix_scaled = scaler.fit_transform(matrix)
        return matrix_scaled

    def concat_train_feature(self):
        train_feature_matrix = self.X_train[self.feature_list].to_numpy()
        val_feature_matrix = self.X_val[self.feature_list].to_numpy()

        # train_feature_matrix_scaled = self.scaled_matrix(train_feature_matrix)
        # val_feature_matrix_scaled = self.scaled_matrix(val_feature_matrix)

        self.train_full_matrix = np.hstack([self.train_text_matrix, train_feature_matrix])
        self.val_full_matrix = np.hstack([self.val_text_matrix, val_feature_matrix])
        print(f"🟢[Concat] train_full_matrix shape: {self.train_full_matrix.shape}")
        print(f"🟢[Concat] val_full_matrix shape: {self.val_full_matrix.shape}")

    def concat_test_feature(self):
        test_feature_matrix = self.test[self.feature_list].to_numpy()
        # test_feature_matrix_scaled = self.scaled_matrix(test_feature_matrix)
        self.test_full_matrix = np.hstack([self.test_text_matrix, test_feature_matrix])
        print(f"🔵[Concat Test] test_full_matrix shape: {self.test_full_matrix.shape}")

    def make_matrix(self):
        self.df_load_train_text_emb()
        self.concat_train_feature()
        self.df_load_test_text_emb()
        self.concat_test_feature()
        if self.tfidf:
            self.concat_tfidf()
            self.log(f"tf-idf ⭕️")
    
    def get_train_matrix(self):
        return self.train_full_matrix, self.val_full_matrix
    
    def get_test_matrix(self):
        return self.test_full_matrix

    def log(self, msg):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, 'a', encoding='utf-8') as f:
            if isinstance(msg, dict):
                for k, v in msg.items():
                    f.write(f"{k}: {v}\n")
            else:
                f.write(str(msg) + "\n")

