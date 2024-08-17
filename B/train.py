import pandas as pd
import os
import numpy as np
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer,TrainingArguments, DataCollatorForSeq2Seq, BitsAndBytesConfig
import torch
print(os.getcwd()) 

# import json
from datasets import load_dataset
from config import DATA_PATH, VAL_SET_SIZE, DATA_SET, MODEL, MODEL_PATH
from utils import generate_prompt, data_collator, save_pretrained, print_trainable_parameters

import os
pwd = os.getcwd()
cur_dir=os.path.dirname(os.path.realpath(__file__))
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
# 加载预训练模型
base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb_config, device_map="auto",torch_dtype=torch.bfloat16)
    
print(print_trainable_parameters(base_model))

from peft import LoraConfig, TaskType, prepare_model_for_kbit_training, get_peft_model
from utils import tokenizer

# 需要修改的lora参数
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
    inference_mode=False, # 训练模式
    bias="none",
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1 # Dropout 比例
)

model = get_peft_model(base_model, config)

print(print_trainable_parameters(model))

#需要调整的训练参数
args = TrainingArguments(
    output_dir="%s/output/%s/%s"%(cur_dir, MODEL, DATA_SET),
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=40, 
    save_steps=100,
    learning_rate=8e-5, # 1e-3 ~ 5e-5
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="tensorboard",
    save_strategy="epoch",  # save checkpoint every epoch
    evaluation_strategy="epoch",
    load_best_model_at_end=True
)

# 读取数据集
data = load_dataset("json", data_files=DATA_PATH)
# 分为训练集，验证集，验证集不是必要
if VAL_SET_SIZE > 0:
    # VAL_SET_SIZE = max(min(VAL_SET_SIZE, int(len(data)/10000)), 1)
    generate_prompt(data["train"][0], is_logger=True)
    train_val = data["train"].train_test_split(test_size=VAL_SET_SIZE, shuffle=True, seed=42)
    train_data = train_val["train"].shuffle().map(generate_prompt)
    val_data = train_val["test"].shuffle().map(generate_prompt)
else:
    generate_prompt(data["train"][0], is_logger=True)
    train_data = data["train"].shuffle().map(generate_prompt)
    val_data = None

trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=data_collator,
)

model.config.use_cache = False

trainer.train()

lora_path='%s/saved_model/%s'%(cur_dir, DATA_SET) # 修改成工作路径

model.config.use_cache = True
trainer.model.save_pretrained(lora_path)
save_pretrained(lora_path)