# -*- coding: utf-8 -*-
from modelscope import AutoTokenizer
from config import MAX_LENGTH_Q, MAX_LENGTH_A, MAX_LENGTH_QA, MODEL, MODEL_PATH
import torch


ID_PAD = 0
ID_BOS = 2
ID_EOS = 1
ID_UNK = 3
ID_MASK = 4
ID_SOT = 106
ID_EOT = 107
ID_BR = 108  # "\n"
ID_USER = 1645
ID_MODEL = 2516


model_path = "Lucachen/gemma2b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)


def generate_prompt(data_point, is_logger=False):
    """   指令微调:
    普通句子续写: bos + text + eos
    带 prompt:
    ‘<start_of_turn>user
    Knock knock.<end_of_turn>
    <start_of_turn>model
    Who’s there?<end_of_turn>model
    <start_of_turn>user
    Gemma.<end_of_turn>
    <start_of_turn>model
    Gemma who?<end_of_turn>model’
    """

    text_system = data_point.get("instruction", "")
    text_input = data_point.get("input", "")
    text_out = data_point.get("output", "")
    prompt_text_0 = "<start_of_turn>system\n{}<end_of_turn>\n"
    prompt_text_1 = "<start_of_turn>user\n{}<end_of_turn>\n"
    prompt_text_2 = "<start_of_turn>model\n{}<end_of_turn>"
    text_0 = prompt_text_0.format(text_system.strip())
    text_1 = prompt_text_1.format(text_input.strip())
    text_2 = prompt_text_2.format(text_out.strip())
    x = tokenizer.encode(text_0 + text_1, add_special_tokens=False)
    y = tokenizer.encode(text_2, add_special_tokens=False)
    if len(x) + len(y) > (MAX_LENGTH_Q + MAX_LENGTH_A):
        x = x[:MAX_LENGTH_Q]
        y = y[:MAX_LENGTH_A]
    x = [ID_BOS] + x
    y = y + [ID_EOS]
    out = {"input_ids": x, "labels": y}
    if is_logger:
        print(text_0)
        print(text_1)
        print(text_2)
        print(out)
    return out


def data_collator(batch):
    # there's probably a way to do this with the tokenizer settings
    len_max_batch = [len(batch[i].get("input_ids")) + len(batch[i].get("labels")) for i in range(len(batch))]
    len_max_batch = min(MAX_LENGTH_QA, max(len_max_batch))
    batch_attention_mask = []
    batch_input_ids = []
    batch_labels = []
    for ba in batch:
        x, y = ba.get("input_ids"), ba.get("labels")
        len_padding = len_max_batch - len(x) - len(y)
        # ## calculate loss of output and input
        if tokenizer.padding_side and tokenizer.padding_side == "left":
            labels = [-100] * len_padding + x + y
            input_ids = [ID_PAD] * len_padding + x + y
            attention_mask = [0] * len_padding + [1] * (len_max_batch - len_padding)
        else:
            labels = x + y + [-100] * len_padding
            input_ids = x + y + [ID_PAD] * len_padding
            attention_mask = [1] * (len(x) + len(y)) + [0] * len_padding
        tensor_attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        tensor_input_ids = torch.tensor(input_ids, dtype=torch.long)
        tensor_labels = torch.tensor(labels, dtype=torch.long)
        batch_attention_mask.append(tensor_attention_mask)
        batch_input_ids.append(tensor_input_ids)
        batch_labels.append(tensor_labels)
    batch_attention_mask = torch.stack(batch_attention_mask)
    batch_input_ids = torch.stack(batch_input_ids)
    batch_labels = torch.stack(batch_labels)
    input_dict = {"attention_mask": batch_attention_mask,  # no use
                  "input_ids": batch_input_ids,
                  "labels": batch_labels,
                  }
    return input_dict


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return f"Trainable model parameters: {trainable_params}\nAll model parameters: {all_param}\nPercentage of trainable model parameters: {100 * trainable_params / all_param:.2f}%"


def save_pretrained(lora_path):
    tokenizer.save_pretrained(lora_path)