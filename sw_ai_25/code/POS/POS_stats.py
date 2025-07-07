import os
import pandas as pd
from collections import Counter
from tqdm.auto import tqdm
from joblib import Parallel, delayed

tqdm.pandas()

# df = pd.DataFrame({
#     'PID': [...],
#     'title': [...],
#     'paragraph_index': [...],
#     'paragraph_pos': [[('형태소1', 'NNG'), ('형태소2', 'JKS'), ...], ...],
#     'generated': [...]
# })


def get_pos_ratio(pos_list, target_tags):
    # 품사별 비율
    total = len(pos_list)
    if total == 0:
        return {tag: 0.0 for tag in target_tags}
    counter = Counter(tag for _, tag in pos_list)
    return {tag: counter.get(tag, 0) / total for tag in target_tags}

def get_special_pos_ratio(pos_list, special_tags=['NNP', 'SL']):
    # 고유 명사(NNP), 외래어(SL) 비율
    total = len(pos_list)
    if total == 0:
        return {tag: 0.0 for tag in special_tags}
    counter = Counter(tag for _, tag in pos_list)
    return {tag: counter.get(tag, 0) / total for tag in special_tags}

def get_josa_eomi_variety(pos_list):
    # 조사/어미 다양성 (종류 수)
    josa_tags = {'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC'}
    eomi_tags = {'EF', 'EC', 'ETN', 'ETM', 'EP'}
    josa_set = set(tag for _, tag in pos_list if tag in josa_tags)
    eomi_set = set(tag for _, tag in pos_list if tag in eomi_tags)
    return {'josa_variety': len(josa_set), 'eomi_variety': len(eomi_set)}

def get_josa_eomi_repeat(pos_list):
    # 조사/어미의 반복 패턴 (동일 조사/어미 연속 사용 빈도)
    josa_tags = {'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC'}
    eomi_tags = {'EF', 'EC', 'ETN', 'ETM', 'EP'}
    prev_josa = None
    prev_eomi = None
    josa_repeat = 0
    eomi_repeat = 0
    for _, tag in pos_list:
        if tag in josa_tags:
            if tag == prev_josa:
                josa_repeat += 1
            prev_josa = tag
        else:
            prev_josa = None
        if tag in eomi_tags:
            if tag == prev_eomi:
                eomi_repeat += 1
            prev_eomi = tag
        else:
            prev_eomi = None
    return {'josa_repeat': josa_repeat, 'eomi_repeat': eomi_repeat}

def get_pos_type_variety(pos_list):
    # 형태소 유형 다양성 (유니크 태그 개수)
    tag_set = set(tag for _, tag in pos_list)
    return {'pos_type_variety': len(tag_set)}

def extract_features_from_paragraph(paragraph_pos):
    features = {}
    main_tags = ['NNG', 'VV', 'JKS', 'JKO', 'EF', 'EC']
    features.update(get_pos_ratio(paragraph_pos, main_tags))
    features.update(get_special_pos_ratio(paragraph_pos))
    features.update(get_josa_eomi_variety(paragraph_pos))
    features.update(get_josa_eomi_repeat(paragraph_pos))
    features.update(get_pos_type_variety(paragraph_pos))
    return features

# --- 병렬 처리 & 진행상황 표시 ---
def add_features_to_df(df, n_jobs=4):
    results = Parallel(n_jobs=n_jobs)(
        delayed(extract_features_from_paragraph)(row) for row in tqdm(df['paragraph_pos'], desc="Feature Extraction")
    )
    feature_df = pd.DataFrame(results, index=df.index) 
    return pd.concat([df, feature_df], axis=1)

def main(prefix, input_dir, idx, output_dir, n_jobs=4):
    data_path=f"../../data/{prefix}_{input_dir}"
    file_name=f"{prefix}_paragraph_pos_{idx}.pkl"
    input_full_name=os.path.join(data_path, file_name)
    
    if not os.path.exists(input_full_name):
        print(f"파일 ❌: {input_full_name}")
        return
    
    df=pd.read_pickle(input_full_name)
    print(f"{file_name} data points: {len(df)} 개")

    df = add_features_to_df(df, n_jobs=n_jobs)

    output_path=f"../../data/{prefix}_{output_dir}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_full_name=os.path.join(output_path, file_name)
    df.to_pickle(output_full_name)
    print(f"[Saved] {output_full_name} 저장 완료 🗂️")
    print(f"{file_name} data points: {len(df)} 개")

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="형 변환")
    parser.add_argument('--prefix', type=str, default="train")
    parser.add_argument('--input_dir', type=str, default="paragraph_pos")
    parser.add_argument('--idx', type=int, required=True)
    parser.add_argument('--output_dir', type=str, default='paragraph_pos_stats')
    parser.add_argument('--n_jobs', type=int, default=4, help='병렬 처리 코어 수')
    args = parser.parse_args()

    main(prefix=args.prefix, 
         input_dir=args.input_dir, 
         idx=args.idx, 
         output_dir=args.output_dir,
         n_jobs=args.n_jobs)
