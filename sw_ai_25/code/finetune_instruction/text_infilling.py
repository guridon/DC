import random
import numpy as np
import torch
import wandb
import math
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType
import evaluate

# seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# config
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
BATCH_SIZE = 2
NUM_EPOCHS = 1
LEARNING_RATE = 2e-5

wandb.init(
    project="Qwen2.5-1.5B-Instruct-infilling_generated1-finetune",
    name="Qwen2.5-1.5B-Instruct-infilling_paragraph",
    config={
        "model_name": MODEL_NAME,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
    }
)

raw_dataset = load_dataset("json", data_files="paragraph_infilling_generated1.jsonl")['train']
split_dataset = raw_dataset.train_test_split(test_size=0.05, seed=SEED)
train_dataset = split_dataset['train']
valid_dataset = split_dataset['test']
print(len(valid_dataset))

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    padding_side="left",
    use_fast=True,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, lora_config)

def preprocess(example):
    instruction = example["instruction"]
    input_text = example["input"]
    output = example["output"]
    text = f"{instruction}\n\n{input_text}\n\n### 답변:\n{output}"
    tokenized = tokenizer(text, truncation=True, padding="longest", max_length=1024)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_train = train_dataset.map(preprocess, batched=False)
tokenized_valid = valid_dataset.map(preprocess, batched=False)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_labels_bleu = [[label] for label in decoded_labels]
    bleu_result = bleu.compute(predictions=decoded_preds, references=decoded_labels_bleu)
    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    meteor_result = meteor.compute(predictions=decoded_preds, references=decoded_labels)
    metrics = {
        "bleu": bleu_result["bleu"],
        "rouge1": rouge_result["rouge1"],
        "rouge2": rouge_result["rouge2"],
        "rougeL": rouge_result["rougeL"],
        "meteor": meteor_result["meteor"],
    }
    return metrics

training_args = TrainingArguments(
    output_dir="./Qwen2.5-1.5B-Instruct-infilling-finetuned-lora",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    eval_accumulation_steps=2,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=10,
    save_steps=500,
    eval_strategy="steps",
    eval_steps=100,
    report_to="wandb",
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    seed=SEED
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

eval_results = trainer.evaluate()
eval_loss = eval_results["eval_loss"]
perplexity = math.exp(eval_loss)
print(f"Validation Loss: {eval_loss:.4f}")
print(f"Validation Perplexity: {perplexity:.2f}")
wandb.log({
    "eval/loss": eval_loss,
    "eval/perplexity": perplexity
})

model.save_pretrained("./Qwen2.5-1.5B-Instruct-infilling-finetuned-lora")
tokenizer.save_pretrained("./Qwen2.5-1.5B-Instruct-infilling-finetuned-lora")
