import numpy as np
import os
import torch
import torch.nn as nn
from datasets import load_dataset, concatenate_datasets

# for vanilla chosen and reject style dataset, such as dendrydong/preference_700K
def build_dataset(data_path, tokenizer, split='train', size=None, model_name=''):
    ds = load_dataset(data_path, split=split)
    
    if size is not None:
        ds = ds.select(range(0, size))

    def formatting_func(example):
        kwargs = {"padding": True, "truncation": True, "max_length": tokenizer.max_length, "return_tensors": "pt"}
        chosen_messages = example['chosen']
        rejected_messages = example['rejected']
        prompt_plus_chosen_response = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        prompt_plus_rejected_response = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)

        if 'GRM' in model_name:
            # add label mask for sft and dpo training
            prompt = example['chosen'][:-1]
            prompt_template = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            tokens_prompt = tokenizer.encode_plus(prompt_template, **kwargs)['input_ids'][0]
            label_chosen = tokens_chosen["input_ids"][0].clone()
            label_chosen[:len(tokens_prompt)] = -100
            label_rejected = tokens_rejected["input_ids"][0].clone()
            label_rejected[:len(tokens_prompt)] = -100
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
                "label_chosen": label_chosen,  'label_rejected': label_rejected
            }
        else:
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
            }

    ds = ds.map(formatting_func, batched=False, num_proc=10) 
    remove_columns = []
    for col in ds.column_names:
        if 'input' not in col and 'attention' not in col and 'label' not in col:
            remove_columns.append(col)
    ds = ds.remove_columns(remove_columns)

    ds.set_format(type="torch")
    return ds


# for UnifiedFeedback
def build_dataset_UF(data_path, tokenizer, split='train', size=None, mode='', model_name=''):
    try:
        ds = load_dataset(data_path, 'all', split=split)
    except:
        ds = load_dataset(data_path, split=split)
    
    # filter data with the same rating
    ds = ds.filter(lambda example: example['conv_A_rating'] != example['conv_B_rating'], num_proc=30)

    if len(mode):
        if mode == '40k' or mode == '40K':
            ds = ds.select(range(0, len(ds), 20)) 
        elif mode == '400k' or mode == '400K':
            ds = ds.select(range(0, len(ds), 2)) 

    if size is not None:
        ds = ds.select(range(0, size))

    def formatting_func(example):
        kwargs = {"padding": True, "truncation": True, "max_length": tokenizer.max_length, "return_tensors": "pt"}
        if example['conv_A_rating'] > example['conv_B_rating']:
            chosen_messages = example['conv_A']
            rejected_messages = example['conv_B']
            margin = example['conv_A_rating'] - example['conv_B_rating']
        else:
            chosen_messages = example['conv_B']
            rejected_messages = example['conv_A']
            margin = example['conv_B_rating'] - example['conv_A_rating']
        
        if 'summarize' in example['source']:
            chosen_messages[0]['content'] = 'Generate one-sentence summary for the following post: ' + chosen_messages[0]['content'].strip()
            rejected_messages[0]['content'] = 'Generate one-sentence summary for the following post: ' + rejected_messages[0]['content'].strip()
        
        prompt_plus_chosen_response = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        prompt_plus_rejected_response = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)
        if 'GRM' in model_name:
            # add label mask for sft and dpo training
            prompt = [example['conv_A'][0]]
            prompt_template = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            tokens_prompt = tokenizer.encode_plus(prompt_template, **kwargs)['input_ids'][0]
            label_chosen = tokens_chosen["input_ids"][0].clone()
            label_chosen[:len(tokens_prompt)] = -100
            label_rejected = tokens_rejected["input_ids"][0].clone()
            label_rejected[:len(tokens_prompt)] = -100
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
                "label_chosen": label_chosen,  'label_rejected': label_rejected,
                # "margin": margin, # GRM does not need this
            }
        else:
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
                "margin": margin, 
            }
        

    ds = ds.map(formatting_func, batched=False, num_proc=10)
    # ds = ds.filter(lambda x: len(x["input_ids_chosen"]) <= script_args.max_length and len(x["input_ids_rejected"]) <= script_args.max_length, num_proc=30)
    remove_columns = []
    for col in ds.column_names:
        if 'input' not in col and 'attention' not in col and 'margin' not in col and 'label' not in col:
            remove_columns.append(col)
    ds = ds.remove_columns(remove_columns)

    ds.set_format(type="torch")
    return ds


# for Skywork Reward Preference 80K
def build_dataset_SK(data_path, tokenizer, split='train', size=None, model_name=''):
    ds = load_dataset(data_path, split=split)

    if size is not None:
        ds = ds.select(range(0, size))

    def formatting_func(example):
        kwargs = {"padding": True, "truncation": True, "max_length": tokenizer.max_length, "return_tensors": "pt"}
        prompt = example['chosen'][0]['content']

        chosen_messages = example['chosen']
        rejected_messages = example['rejected']

        prompt_plus_chosen_response = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        prompt_plus_rejected_response = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)
        if 'GRM' in model_name:
            # add label mask for sft and dpo
            prompt_template = tokenizer.apply_chat_template([{"content": prompt, "role": "user" }], tokenize=False, add_generation_prompt=True)
            tokens_prompt = tokenizer.encode_plus(prompt_template, **kwargs)['input_ids'][0]
            label_chosen = tokens_chosen["input_ids"][0].clone()
            label_chosen[:len(tokens_prompt)] = -100
            label_rejected = tokens_rejected["input_ids"][0].clone()
            label_rejected[:len(tokens_prompt)] = -100
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
                "label_chosen": label_chosen,  'label_rejected': label_rejected
            }
        else:
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
            }

    ds = ds.map(formatting_func, batched=False, num_proc=10) 
    ds.set_format(type="torch")
    return ds


def load_train_eval_dataset(data_path, tokenizer, size=None, mode='', model_name=''):
    if 'Unified' in data_path:
        # mode is only used for loading training data
        train_dataset = build_dataset_UF(data_path, tokenizer, split='train', size=size, mode=mode, model_name=model_name) 
        eval_dataset = build_dataset_UF(data_path, tokenizer, split='val', model_name=model_name)
    elif 'Skywork' in data_path:
        dataset = build_dataset_SK(data_path, tokenizer, split='train', size=size, model_name=model_name)
        dataset_split = dataset.train_test_split(test_size=0.005)
        train_dataset, eval_dataset = dataset_split['train'], dataset_split['test']
    else:
        dataset = build_dataset(data_path, tokenizer, split='train', size=size, model_name=model_name) 
        dataset_split = dataset.train_test_split(test_size=0.01)
        train_dataset, eval_dataset = dataset_split['train'], dataset_split['test']
    return train_dataset, eval_dataset




# In load_datasets.py (or similar)

def build_dataset_helpsteer(data_path, tokenizer, split='train', size=None):
    """
    Build a dataset for nvidia/HelpSteer2 where each sample contains:
      - a prompt and response (concatenated as a single text input)
      - five attribute scores: helpfulness, correctness, coherence, complexity, verbosity.
    The processing follows a similar structure to the regression dataset.
    """
    ds = load_dataset(data_path, split=split)
    if size is not None:
        ds = ds.select(range(0, size))
    
    # For each sample, concatenate prompt and response, then fetch target attribute scores.
    def formatting_func(example):
        text = example["prompt"] + " " + example["response"]
        kwargs = {
            "padding": True,
            "truncation": True,
            "max_length": tokenizer.max_length,
            "return_tensors": "pt"
        }
        tokens = tokenizer.encode_plus(text, **kwargs)
        target_attributes = [
            float(example["helpfulness"]),
            float(example["correctness"]),
            float(example["coherence"]),
            float(example["complexity"]),
            float(example["verbosity"]),
        ]
        return {
            "input_ids_reg": tokens["input_ids"][0],
            "attention_mask_reg": tokens["attention_mask"][0],
            "target_attributes": torch.tensor(target_attributes, dtype=torch.float),
        }
    
    ds = ds.map(formatting_func, batched=False)
    
    # Remove any columns not needed for training.
    remove_columns = [col for col in ds.column_names if col not in [
        "input_ids_reg", "attention_mask_reg", "target_attributes"
    ]]
    ds = ds.remove_columns(remove_columns)
    ds.set_format(type="torch")
    return ds




from torch.utils.data import Dataset

class CombinedDataset(Dataset):
    def __init__(self, skywork_ds, helpsteer_ds):
        self.skywork_ds = skywork_ds
        self.helpsteer_ds = helpsteer_ds

    def __len__(self):
        return len(self.skywork_ds)

    def __getitem__(self, idx):
        # Skywork sample is taken directly.
        skywork_sample = self.skywork_ds[idx]
        # For helpsteer, cycle through if necessary.
        helpsteer_idx = idx % len(self.helpsteer_ds)
        helpsteer_sample = self.helpsteer_ds[helpsteer_idx]
        # Mark sample type in each part:
        skywork_sample["sample_type"] = 0  # for skywork (it will yield two items downstream)
        helpsteer_sample["sample_type"] = 1
        return {
            "skywork": skywork_sample,
            "helpsteer": helpsteer_sample
        }


def load_train_eval_combined_dataset(skywork_path, helpsteer_path, tokenizer, size=None, mode='', model_name=''):
    # Load skywork dataset using your existing build_dataset_SK
    skywork_ds = build_dataset_SK(skywork_path, tokenizer, split='train', size=size, model_name=model_name)
    # For evaluation, you might only use skywork or handle separately.
    helpsteer_ds = build_dataset_helpsteer(helpsteer_path, tokenizer, split='train')
    combined_ds = CombinedDataset(skywork_ds, helpsteer_ds)
    return combined_ds  # use this combined dataset for training


def build_dataset_bt(data_path, tokenizer, split='train', size=None):
    ds = load_dataset(data_path, split=split)
    if size is not None:
        ds = ds.select(range(0, size))
    
    def formatting_func(example):
        kwargs = {
            "padding": True, 
            "truncation": True, 
            "max_length": tokenizer.max_length, 
            "return_tensors": "pt"
        }
        # Extract the prompt.
        prompt = example["prompt"]
        # For chosen and rejected, concatenate the "content" fields of all messages.
        chosen_messages = example["chosen"]
        rejected_messages = example["rejected"]
        chosen_text = " ".join([msg["content"] for msg in chosen_messages])
        rejected_text = " ".join([msg["content"] for msg in rejected_messages])
        
        # Concatenate prompt with the chosen/rejected texts.
        text_chosen = prompt + " " + chosen_text
        text_rejected = prompt + " " + rejected_text
        
        tokens_chosen = tokenizer.encode_plus(text_chosen, **kwargs)
        tokens_rejected = tokenizer.encode_plus(text_rejected, **kwargs)
        return {
            "input_ids_chosen": tokens_chosen["input_ids"][0],
            "attention_mask_chosen": tokens_chosen["attention_mask"][0],
            "input_ids_rejected": tokens_rejected["input_ids"][0],
            "attention_mask_rejected": tokens_rejected["attention_mask"][0],
        }
    
    ds = ds.map(formatting_func, batched=False)
    remove_columns = [col for col in ds.column_names if col not in [
        "input_ids_chosen", "attention_mask_chosen", "input_ids_rejected", "attention_mask_rejected"
    ]]
    ds = ds.remove_columns(remove_columns)
    ds.set_format(type="torch")
    return ds


from datasets import load_dataset, Dataset as HFDataset
import torch

def build_dataset_regression(data_path, tokenizer, split='train', size=None):
    ds = load_dataset(data_path, split=split)
    # ds = ds.select(range(0, 100))
    if size is not None:
        ds = ds.select(range(0, size))
    
    # Define the flattening function to process completions.
    def flatten_func(example):
        samples = []
        instruction = example["instruction"]
        for comp in example["completions"]:
            response = comp["response"]
            try:
                instr_follow = float(comp['annotations']['instruction_following']['Rating'])
                honesty = float(comp['annotations']['honesty']['Rating'])
                truthfulness = float(comp['annotations']['truthfulness']['Rating'])
                helpfulness = float(comp['annotations']['helpfulness']['Rating'])
            except Exception as e:
                instr_follow = honesty = truthfulness = helpfulness = 0.0
            samples.append({
                "text": instruction + " " + response,
                "target_attributes": [instr_follow, honesty, truthfulness, helpfulness],
            })
        return samples

    # Manually flatten the dataset.
    all_samples = []
    for example in ds:
        all_samples.extend(flatten_func(example))
    
    # Create a new Dataset from the flattened list.
    ds = HFDataset.from_list(all_samples)
    
    def formatting_func(example):
        kwargs = {
            "padding": True, 
            "truncation": True, 
            "max_length": tokenizer.max_length, 
            "return_tensors": "pt"
        }
        tokens = tokenizer.encode_plus(example["text"], **kwargs)
        return {
            "input_ids_reg": tokens["input_ids"][0],
            "attention_mask_reg": tokens["attention_mask"][0],
            "target_attributes": torch.tensor(example["target_attributes"], dtype=torch.float),
        }
    
    ds = ds.map(formatting_func, batched=False)
    remove_columns = [col for col in ds.column_names if col not in [
        "input_ids_reg", "attention_mask_reg", "target_attributes"
    ]]
    ds = ds.remove_columns(remove_columns)
    ds.set_format(type="torch")
    return ds



# --- Combined Dataset class ---
from torch.utils.data import Dataset

# class CombinedDataset(Dataset):
#     def __init__(self, bt_ds, reg_ds):
#         self.bt_ds = bt_ds
#         self.reg_ds = reg_ds

#     def __len__(self):
#         # You can decide the length; here we cycle the regression samples if needed.
#         return len(self.bt_ds)

#     def __getitem__(self, idx):
#         bt_sample = self.bt_ds[idx]
#         reg_idx = idx % len(self.reg_ds)
#         reg_sample = self.reg_ds[reg_idx]
#         # Mark sample types: 0 for BT and 1 for regression.
#         bt_sample["sample_type"] = 0
#         reg_sample["sample_type"] = 1
#         return {"bt": bt_sample, "reg": reg_sample}
from torch.utils.data import Dataset

class CombinedDataset(Dataset):
    def __init__(self, bt_ds, reg_ds):
        self.bt_ds = bt_ds
        self.reg_ds = reg_ds
        # Use the maximum length of both datasets.
        self.length = max(len(bt_ds), len(reg_ds))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Cycle each dataset independently.
        bt_sample = self.bt_ds[idx % len(self.bt_ds)]
        reg_sample = self.reg_ds[idx % len(self.reg_ds)]
        # Mark sample types: 0 for BT and 1 for regression.
        bt_sample["sample_type"] = 0
        reg_sample["sample_type"] = 1
        return {"bt": bt_sample, "reg": reg_sample}
