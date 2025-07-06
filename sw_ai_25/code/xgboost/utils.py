

def pick_list(df_columns):
    train_emb_list=["paragraph_text_emb","title_emb"]
    test_emb_list=["paragraph_text_emb", "title_emb"]

    for idx, col in enumerate(df_columns):
        print(f"{idx}: {col}")
    selected = input("사용할 feature의 번호를 콤마(,)로 구분해서 입력 (ex: 0,2,3): ")
    selected_idx= [int(x.strip()) for x in selected.split(",") if x.strip().isdigit()]
    feature_list = [df_columns[i] for i in selected_idx]
    return train_emb_list, test_emb_list, feature_list
