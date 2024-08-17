import pandas as pd
import json

DATA_PATH = "/home/aiservice/workspace/data/questionBData" # 填入考试环境的数据集路径

pre_data_path = "/home/aiservice/workspace/questionB/pre_data"

def qqp_process():
    train_df = pd.read_csv('%s/QQP/QQP_train.tsv'%(DATA_PATH), sep='\t', on_bad_lines="skip", header=0)
    dev_df = pd.read_csv('%s/QQP/QQP_dev.tsv'%(DATA_PATH), sep='\t', on_bad_lines = "skip", header=0)

    train_res = []

    for i, row in train_df.iterrows():
        if i < 4000:
            if len(row) == 6:
                # 构造每一项
                if type(row['question2']) == float:
                    print(1)
                    continue
                tmp = {}
                tmp['instruction'] = "Are the scentences meaning the same ? same output 1 , otherwise 0 ." # 构造合适的 instruction 
                tmp['input'] = row['question1'] + ' ' + row['question2']
                tmp['output'] = str(row['is_duplicate'])
                train_res.append(tmp)

    dev_res = []

    for i, row in dev_df.iterrows():
        if i < 150:
            if len(row) == 5:
                if type(row['question2']) == float:
                    continue
                # 构造每一项
                tmp = {}
                tmp['instruction'] = "Are the scentences meaning the same ? same output 1 , otherwise 0 ." # 构造合适的 instruction 
                tmp['input'] = row['question1'] + ' ' + row['question2']
                dev_res.append(tmp)

    json_str = json.dumps(train_res, indent=2, ensure_ascii=False)   
    with open('%s/QQP/QQP_train.json'%(pre_data_path),"w", encoding='utf-8') as f:
        f.write(json_str)
        f.close()

    json_str = json.dumps(dev_res, indent=2, ensure_ascii=False)
    with open('%s/QQP/QQP_dev.json'%(pre_data_path), "w", encoding='utf-8') as f:
        f.write(json_str)
        f.close()

def cola_process():
    train_df = pd.read_csv('%s/CoLA/CoLA_train.tsv'%(DATA_PATH), sep='\t', on_bad_lines="skip", header=None)
    dev_df = pd.read_csv('%s/CoLA/CoLA_dev.tsv'%(DATA_PATH), sep='\t', on_bad_lines = "skip", header=None)

    train_res = []

    for i, row in train_df.iterrows():
        if i < 4000:
            if len(row) == 4:
                # 构造每一项
                if type(row[3]) == float:
                    print(1)
                    continue
                tmp = {}
                tmp['instruction'] = "Is the grammar of the scentence right ? right output 1 , otherwise 0 ." # 构造合适的 instruction 
                # tmp['instruction'] = ''
                tmp['input'] = row[3]
                tmp['output'] = str(row[1])
                train_res.append(tmp)

    dev_res = []

    for i, row in dev_df.iterrows():
        if i < 150:
            if len(row) == 3:
                if type(row[2]) == float:
                    continue
                # 构造每一项
                tmp = {}
                tmp['instruction'] = "Is the grammar of the scentence right ? right output 1 , otherwise 0 ." # 构造合适的 instruction 
                # tmp['instruction'] = ''
                tmp['input'] = row[2]
                dev_res.append(tmp)

    json_str = json.dumps(train_res, indent=2, ensure_ascii=False)   
    with open('%s/CoLA/CoLA_train.json'%(pre_data_path),"w", encoding='utf-8') as f:
        f.write(json_str)
        f.close()

    json_str = json.dumps(dev_res, indent=2, ensure_ascii=False)
    with open('%s/CoLA/CoLA_dev.json'%(pre_data_path), "w", encoding='utf-8') as f:
        f.write(json_str)
        f.close()

qqp_process()
cola_process()