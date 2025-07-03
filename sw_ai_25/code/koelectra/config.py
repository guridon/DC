import torch
from dataclasses import dataclass

@dataclass
class Config:
    model_name : str = "monologg/koelectra-base-v3-discriminator"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir: str = "../../data/"
    train_file: str = ""
    test_file: str = ""
    output_dir : str = "./results"
    num_labels : int = 2

@dataclass
class TrainingConfig:
    output_dir: str = "./models/checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-4
    warmup_steps: int = 100
    max_steps: int = 1000
    logging_steps: int = 25
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3
    fp16: bool = True
    
    use_wandb: bool = True
    project_name: str = None
    run_name: str = None