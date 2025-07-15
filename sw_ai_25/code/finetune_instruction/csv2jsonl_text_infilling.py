import pandas as pd
import random
import json

df = pd.read_csv('train_paragraph_id.csv')
df = df[(df['generated'] == 1) & (df['paragraph_index'] <= 40)].reset_index(drop=True)

result = []
random.seed(42)

for title, group in df.groupby('title'):
    group_sorted = group.sort_values('paragraph_index')
    paragraphs = group_sorted['paragraph_text'].tolist()
    pids = group_sorted['PID'].tolist()
    n = len(paragraphs)
    
    if n <= 1:
        continue 
    if n <= 5:
        num_outputs = 1
    elif n <= 10:
        num_outputs = 2
    else:
        num_outputs = 3

    output_indices = random.sample(range(n), min(num_outputs, n))

    for idx in output_indices:
        output_para = paragraphs[idx]
        output_pid = pids[idx]
        input_paras = [p for i, p in enumerate(paragraphs) if i != idx]
        input_text = '\n'.join(input_paras)
        prompt = f"'{title}'라는 주제로 작성된 글에서 빠진 문단을 자연스럽게 이어서 써주세요."
        result.append({
            "PID": output_pid,
            "title": title,
            "instruction": prompt,
            "input": input_text,
            "output": output_para
        })

with open('paragraph_infilling_generated1.jsonl', 'w', encoding='utf-8') as f:
    for rec in result:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
