# -*- coding: utf-8 -*-
import pandas as pd
import json


DATA_PATH = "./data"  # 填入考试环境的数据集路径
pre_data_path = "./pre_data"


def process_qqp(balance: bool = False):
    train_df = pd.read_csv('%s/QQP/QQP_train.tsv' % DATA_PATH, sep='\t', on_bad_lines="skip", header=0)
    dev_df = pd.read_csv('%s/QQP/QQP_dev.tsv' % DATA_PATH, sep='\t', on_bad_lines="skip", header=0)
    train_df = train_df.dropna(subset=["question2"])
    dev_df = dev_df.dropna(subset=["question2"])
    train_df["sentence1"] = train_df["question1"]
    train_df["sentence2"] = train_df["question2"]
    train_df["labels"] = train_df["is_duplicate"].astype(int)
    dev_df["sentence1"] = dev_df["question1"]
    dev_df["sentence2"] = dev_df["question2"]
    if balance:
        train_df_pos = train_df[train_df["labels"] == 1]
        train_df_neg = train_df[train_df["labels"] == 0].head(train_df_pos.shape[0])
        train_df = pd.concat([train_df_pos, train_df_neg]).sample(frac=1, random_state=42)
    train_df = train_df[["sentence1", "sentence2", "labels"]].sample(min(4000, train_df.shape[0]), random_state=42)
    dev_df = dev_df[["sentence1", "sentence2"]].head(150)
    train_json = train_df.to_json(orient="records")
    dev_json = dev_df.to_json(orient="records")
    with open('%s/QQP/QQP_train.json' % pre_data_path, "w", encoding='utf-8') as f:
        f.write(train_json)
    with open('%s/QQP/QQP_dev.json' % pre_data_path, "w", encoding='utf-8') as f:
        f.write(dev_json)


def qqp_process():
    train_df = pd.read_csv('%s/QQP/QQP_train.tsv' % DATA_PATH, sep='\t', on_bad_lines="skip", header=0)
    dev_df = pd.read_csv('%s/QQP/QQP_dev.tsv' % DATA_PATH, sep='\t', on_bad_lines="skip", header=0)

    train_res = []
    for i, row in train_df.iterrows():
        if i < 4000:
            if len(row) == 6:
                # 构造每一项
                if type(row['question2']) is float:
                    continue
                tmp = {}
                tmp['instruction'] = "Are the sentences meaning the same ? same output 1 , otherwise 0 ."  # 构造合适的 instruction
                tmp['input'] = row['question1'] + ' ' + row['question2']
                tmp['output'] = str(row['is_duplicate'])
                train_res.append(tmp)
        else:
            break

    dev_res = []
    for i, row in dev_df.iterrows():
        if i < 150:
            if len(row) == 5:
                if type(row['question2']) is float:
                    continue
                # 构造每一项
                tmp = {}
                tmp['instruction'] = "Are the sentences meaning the same ? same output 1 , otherwise 0 ."  # 构造合适的 instruction
                tmp['input'] = row['question1'] + ' ' + row['question2']
                dev_res.append(tmp)
        else:
            break

    json_str = json.dumps(train_res, indent=2, ensure_ascii=False)
    with open('%s/QQP/QQP_train.json' % pre_data_path, "w", encoding='utf-8') as f:
        f.write(json_str)

    json_str = json.dumps(dev_res, indent=2, ensure_ascii=False)
    with open('%s/QQP/QQP_dev.json' % pre_data_path, "w", encoding='utf-8') as f:
        f.write(json_str)


def process_cola(balance: bool = False):
    train_df = pd.read_csv('%s/CoLA/CoLA_train.tsv' % DATA_PATH, sep='\t', on_bad_lines="skip", header=None)
    dev_df = pd.read_csv('%s/CoLA/CoLA_dev.tsv' % DATA_PATH, sep='\t', on_bad_lines="skip", header=None)
    train_df = train_df.dropna(subset=[3])
    dev_df = dev_df.dropna(subset=[2])
    train_df["sentence"] = train_df[3]
    train_df["labels"] = train_df[1].astype(int)
    dev_df["sentence"] = dev_df[2]
    if balance:
        train_df_neg = train_df[train_df["labels"] == 0]
        train_df_pos = train_df[train_df["labels"] == 1].head(train_df_neg.shape[0])
        train_df = pd.concat([train_df_pos, train_df_neg]).sample(frac=1, random_state=42)
    train_df = train_df[["sentence", "labels"]].sample(min(4000, train_df.shape[0]), random_state=42)
    dev_df = dev_df[["sentence"]].head(150)
    train_json = train_df.to_json(orient="records")
    dev_json = dev_df.to_json(orient="records")
    with open('%s/CoLA/CoLA_train.json' % pre_data_path, "w", encoding='utf-8') as f:
        f.write(train_json)
    with open('%s/CoLA/CoLA_dev.json' % pre_data_path, "w", encoding='utf-8') as f:
        f.write(dev_json)


def cola_process():
    train_df = pd.read_csv('%s/CoLA/CoLA_train.tsv' % DATA_PATH, sep='\t', on_bad_lines="skip", header=None)
    dev_df = pd.read_csv('%s/CoLA/CoLA_dev.tsv' % DATA_PATH, sep='\t', on_bad_lines="skip", header=None)

    train_res = []
    for i, row in train_df.iterrows():
        if i < 4000:
            if len(row) == 4:
                # 构造每一项
                if type(row[3]) is float:
                    continue
                tmp = {}
                tmp['instruction'] = "Is the grammar of the sentence right ? right output 1 , otherwise 0 ."  # 构造合适的 instruction
                tmp['input'] = row[3]
                tmp['output'] = str(row[1])
                train_res.append(tmp)
        else:
            break

    dev_res = []
    for i, row in dev_df.iterrows():
        if i < 150:
            if len(row) == 2:
                if type(row[1]) is float:
                    continue
                # 构造每一项
                tmp = {}
                tmp['instruction'] = "Is the grammar of the sentence right ? right output 1 , otherwise 0 ."  # 构造合适的 instruction
                tmp['input'] = row[1]
                dev_res.append(tmp)
        else:
            break

    json_str = json.dumps(train_res, indent=2, ensure_ascii=False)
    with open('%s/CoLA/CoLA_train.json' % pre_data_path, "w", encoding='utf-8') as f:
        f.write(json_str)

    json_str = json.dumps(dev_res, indent=2, ensure_ascii=False)
    with open('%s/CoLA/CoLA_dev.json' % pre_data_path, "w", encoding='utf-8') as f:
        f.write(json_str)


def process_sst(balance: bool = False):
    train_df = pd.read_csv('%s/SST-2/SST-2_train.tsv' % DATA_PATH, sep='\t', on_bad_lines="skip", header=0)
    dev_df = pd.read_csv('%s/SST-2/SST-2_dev.tsv' % DATA_PATH, sep='\t', on_bad_lines="skip", header=0)
    train_df["labels"] = train_df["label"].astype(int)
    if balance:
        train_df_neg = train_df[train_df["labels"] == 0]
        train_df_pos = train_df[train_df["labels"] == 1].head(train_df_neg.shape[0])
        train_df = pd.concat([train_df_pos, train_df_neg]).sample(frac=1, random_state=42)
    train_df = train_df[["sentence", "labels"]].sample(min(4000, train_df.shape[0]), random_state=42)
    dev_df = dev_df[["sentence"]].head(150)
    train_json = train_df.to_json(orient="records")
    dev_json = dev_df.to_json(orient="records")
    with open('%s/SST-2/SST-2_train.json' % pre_data_path, "w", encoding='utf-8') as f:
        f.write(train_json)
    with open('%s/SST-2/SST-2_dev.json' % pre_data_path, "w", encoding='utf-8') as f:
        f.write(dev_json)


def sst_process():
    train_df = pd.read_csv('%s/SST-2/SST-2_train.tsv' % DATA_PATH, sep='\t', on_bad_lines="skip", header=0)
    dev_df = pd.read_csv('%s/SST-2/SST-2_dev.tsv' % DATA_PATH, sep='\t', on_bad_lines="skip", header=0)

    train_res = []
    for i, row in train_df.iterrows():
        if i < 4000:
            if len(row) == 2:
                # 构造每一项
                tmp = {}
                tmp['instruction'] = "Is the sentence positive ? positive output 1 , otherwise 0 ."  # 构造合适的 instruction
                tmp['input'] = row['sentence']
                tmp['output'] = str(row["label"])
                train_res.append(tmp)
        else:
            break

    dev_res = []
    for i, row in dev_df.iterrows():
        if i < 150:
            if len(row) == 1:
                # 构造每一项
                tmp = {}
                tmp['instruction'] = "Is the sentence positive ? positive output 1 , otherwise 0 ."  # 构造合适的 instruction
                tmp['input'] = row['sentence']
                dev_res.append(tmp)
        else:
            break

    json_str = json.dumps(train_res, indent=2, ensure_ascii=False)
    with open('%s/SST-2/SST-2_train.json' % pre_data_path, "w", encoding='utf-8') as f:
        f.write(json_str)

    json_str = json.dumps(dev_res, indent=2, ensure_ascii=False)
    with open('%s/SST-2/SST-2_dev.json' % pre_data_path, "w", encoding='utf-8') as f:
        f.write(json_str)


def process_mrpc(balance: bool = False):
    train_df = pd.read_csv('%s/MRPC/MRPC_train.tsv' % DATA_PATH, sep='\t', on_bad_lines="skip", header=0)
    dev_df = pd.read_csv('%s/MRPC/MRPC_dev.tsv' % DATA_PATH, sep='\t', on_bad_lines="skip", header=0)
    train_df["sentence1"] = train_df["#1 String"]
    train_df["sentence2"] = train_df["#2 String"]
    train_df["labels"] = train_df["Quality"].astype(int)
    dev_df["sentence1"] = dev_df["#1 String"]
    dev_df["sentence2"] = train_df["#2 String"]
    if balance:
        train_df_neg = train_df[train_df["labels"] == 0]
        train_df_pos = train_df[train_df["labels"] == 1].head(train_df_neg.shape[0])
        train_df = pd.concat([train_df_pos, train_df_neg]).sample(frac=1, random_state=42)
    train_df = train_df[["sentence1", "sentence2", "labels"]].sample(min(4076, train_df.shape[0]), random_state=42)
    dev_df = dev_df[["sentence1", "sentence2"]].head(1725)
    train_json = train_df.to_json(orient="records")
    dev_json = dev_df.to_json(orient="records")
    with open('%s/MRPC/MRPC_train.json' % pre_data_path, "w", encoding='utf-8') as f:
        f.write(train_json)
    with open('%s/MRPC/MRPC_dev.json' % pre_data_path, "w", encoding='utf-8') as f:
        f.write(dev_json)


def mrpc_process():
    train_df = pd.read_csv('%s/MRPC/MRPC_train.tsv' % DATA_PATH, sep='\t', on_bad_lines="skip", header=0)
    dev_df = pd.read_csv('%s/MRPC/MRPC_dev.tsv' % DATA_PATH, sep='\t', on_bad_lines="skip", header=0)

    train_res = []
    for i, row in train_df.iterrows():
        if i < 4000:
            if len(row) == 5:
                # 构造每一项
                tmp = {}
                tmp['instruction'] = "Are the sentences meaning the same ? same output 1 , otherwise 0 ."  # 构造合适的 instruction
                tmp['input'] = row['#1 String'] + " " + row["#2 String"]
                tmp['output'] = str(row["Quality"])
                train_res.append(tmp)
        else:
            break

    dev_res = []
    for i, row in dev_df.iterrows():
        if i < 150:
            if len(row) == 5:
                # 构造每一项
                tmp = {}
                tmp['instruction'] = "Are the sentences meaning the same ? same output 1 , otherwise 0 ."  # 构造合适的 instruction
                tmp['input'] = row['#1 String'] + " " + row["#2 String"]
                dev_res.append(tmp)
        else:
            break

    json_str = json.dumps(train_res, indent=2, ensure_ascii=False)
    with open('%s/MRPC/MRPC_train.json' % pre_data_path, "w", encoding='utf-8') as f:
        f.write(json_str)

    json_str = json.dumps(dev_res, indent=2, ensure_ascii=False)
    with open('%s/MRPC/MRPC_dev.json' % pre_data_path, "w", encoding='utf-8') as f:
        f.write(json_str)


def process_rte(balance: bool = False):
    train_df = pd.read_csv('%s/RTE/RTE_train.tsv' % DATA_PATH, sep='\t', on_bad_lines="skip", header=0)
    dev_df = pd.read_csv('%s/RTE/RTE_dev.tsv' % DATA_PATH, sep='\t', on_bad_lines="skip", header=0)
    train_df["labels"] = train_df["label"].map({
        "entailment": 1,
        "not_entailment": 0
    })
    dev_df["labels"] = dev_df["label"].map({
        "entailment": 1,
        "not_entailment": 0
    })
    if balance:
        train_df_neg = train_df[train_df["labels"] == 0]
        train_df_pos = train_df[train_df["labels"] == 1].head(train_df_neg.shape[0])
        train_df = pd.concat([train_df_pos, train_df_neg]).sample(frac=1, random_state=42)
    train_df = train_df[["sentence1", "sentence2", "labels"]].sample(min(2490, train_df.shape[0]), random_state=42)
    dev_df = dev_df[["sentence1", "sentence2"]].head(277)
    train_json = train_df.to_json(orient="records")
    dev_json = dev_df.to_json(orient="records")
    with open('%s/RTE/RTE_train.json' % pre_data_path, "w", encoding='utf-8') as f:
        f.write(train_json)
    with open('%s/RTE/RTE_dev.json' % pre_data_path, "w", encoding='utf-8') as f:
        f.write(dev_json)


def rte_process():
    train_df = pd.read_csv('%s/RTE/RTE_train.tsv' % DATA_PATH, sep='\t', on_bad_lines="skip", header=0)
    dev_df = pd.read_csv('%s/RTE/RTE_dev.tsv' % DATA_PATH, sep='\t', on_bad_lines="skip", header=0)

    train_res = []
    for i, row in train_df.iterrows():
        if i < 4000:
            if len(row) == 4:
                tmp = {}
                tmp['instruction'] = "Is sentence 1 entail sentence 2 ? entailment or not_entailment"  # 构造合适的 instruction
                tmp['input'] = row['sentence1'] + ' ' + row['sentence2']
                tmp['output'] = str(row['label'])
                train_res.append(tmp)
        else:
            break

    dev_res = []
    for i, row in dev_df.iterrows():
        if i < 150:
            if len(row) == 4:
                # 构造每一项
                tmp = {}
                tmp['instruction'] = "Is sentence 1 entail sentence 2 ? entailment or not_entailment"  # 构造合适的 instruction
                tmp['input'] = row['sentence1'] + ' ' + row['sentence2']
                dev_res.append(tmp)
        else:
            break

    json_str = json.dumps(train_res, indent=2, ensure_ascii=False)
    with open('%s/RTE/RTE_train.json' % pre_data_path, "w", encoding='utf-8') as f:
        f.write(json_str)

    json_str = json.dumps(dev_res, indent=2, ensure_ascii=False)
    with open('%s/RTE/RTE_dev.json' % pre_data_path, "w", encoding='utf-8') as f:
        f.write(json_str)


if __name__ == "__main__":
    # qqp_process()
    # cola_process()
    # sst_process()
    process_qqp(True)
    process_cola(True)
    process_sst(True)
