import pandas as pd
import os

from transformers import AutoModelForSequenceClassification
from config import Config


class Loader:
    def __init__(self, config: Config):
        self.config=config
    
    def load_model(self):
        return AutoModelForSequenceClassification.from_pretrained(
        self.config.model_name,
        num_labels=self.config.num_labels
    )

    def load_dataet(self, file_name: str):
        train_data_path=os.path.join(self.config.data_dir,file_name)
        train_df=pd.read_csv(train_data_path)

        
