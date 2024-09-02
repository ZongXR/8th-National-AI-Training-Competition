# -*- coding: utf-8 -*-
import os
from peft import LoraConfig, TaskType, get_peft_model
from datasets import load_dataset
import torch
from transformers import Trainer, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, GemmaForSequenceClassification
from transformers.trainer_utils import EvalPrediction
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from config import DATA_PATH, VAL_SET_SIZE, DATA_SET, MODEL, MODEL_PATH


cur_dir = os.path.dirname(os.path.realpath(__file__))


def compute_metrics(pred: EvalPrediction):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    accuracy = accuracy_score(labels, preds)
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def train():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    # 加载模型
    model = GemmaForSequenceClassification.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        num_labels=2
    )

    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        # 任务
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora_config)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.padding_side = 'right'

    # 加载数据集
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    if DATA_SET in ("QQP", "MRPC", "RTE"):
        dataset = dataset.map(lambda x: tokenizer(x["sentence1"], x["sentence2"], truncation=True, padding=True), batched=True)
    else:
        dataset = dataset.map(lambda x: tokenizer(x["sentence"], truncation=True, padding=True), batched=True)

    if VAL_SET_SIZE > 0:
        train_val = dataset.train_test_split(test_size=VAL_SET_SIZE, shuffle=True, seed=42)
        train_data = train_val["train"]
        val_data = train_val["test"]
    else:
        train_data = dataset.shuffle(seed=42)
        val_data = None

    args = TrainingArguments(
        output_dir="%s/output/%s/%s" % (cur_dir, MODEL, DATA_SET),  # directory to save and repository id
        do_train=True,
        do_eval=True,
        num_train_epochs=100,  # number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,  # number of steps before performing a backward/update pass
        gradient_checkpointing=True,  # use gradient checkpointing to save memory
        optim="adamw_torch_fused",  # use fused adamw optimizer
        logging_steps=10,  # log every 10 steps
        learning_rate=1e-4,  # learning rate, based on QLoRA paper
        max_grad_norm=1.0,  # max gradient norm based on QLoRA paper
        warmup_ratio=0.0,  # warmup ratio based on QLoRA paper
        lr_scheduler_type="reduce_lr_on_plateau",  # use constant learning rate scheduler
        report_to="tensorboard",
        save_strategy="epoch",  # save checkpoint every epoch
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        seed=42,
        data_seed=42,
        dataloader_num_workers=10,
        dataloader_persistent_workers=True
    )

    # max sequence length for model and packing of the dataset
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    lora_path = '%s/saved_model/%s' % (cur_dir, DATA_SET)  # 修改成工作路径
    trainer.save_model(lora_path)


if __name__ == '__main__':
    train()
