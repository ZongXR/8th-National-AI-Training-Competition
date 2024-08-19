import os
from modelscope import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from datasets import load_dataset
from peft import PeftModel, LoraConfig, TaskType
from transformers import pipeline
import pandas as pd
import json
from config import DATA_SET
cur_dir=os.path.dirname(os.path.realpath(__file__))
import sys

lora_path ="%s/saved_model/%s"%(cur_dir, DATA_SET)
# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(lora_path)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(lora_path, device_map="auto",torch_dtype=torch.bfloat16)

# 默认参数
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)


# 读取测试数据集
# DATA_PATH = "%s/%s_"%(evaldata_dir, DATA_SET1) #无需修改
test_data = load_dataset("json", data_files="%s/pre_data/%s/%s_dev.json"%(cur_dir, DATA_SET, DATA_SET)) # 
# test文件基本没有标签无法评判模型精度，所以一般使用dev.json

# 开始测试
total_number = len(test_data['train'])
right_number = 0
device = "cuda:0"

f_out = open("%s/result_%s.txt"%(cur_dir,DATA_SET), 'w', encoding='utf-8')

for i in range(0, total_number):
    messages = [
        {"role": "user", "content": test_data['train'][i]['instruction']+ " " + test_data['train'][i]['input']} # role不需要调整 
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=64
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # print(response)
    if i == total_number-1:
        f_out.write(response[0])
    else:
        f_out.write(response[0]+'\n')
    f_out.flush()
