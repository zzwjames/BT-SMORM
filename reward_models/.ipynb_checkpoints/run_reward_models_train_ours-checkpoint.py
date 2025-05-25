from dataclasses import dataclass, field
from typing import List, Optional
from accelerate import Accelerator
import evaluate
import numpy as np
import os
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
from reward_trainer import SimpleRewardTrainer, RewardDataCollatorWithPadding
from load_datasets import load_train_eval_dataset
from utils import print_trainable_parameters, compute_metrics, freeze_trainable_parameters


@dataclass
class ScriptArguments:
    # training args
    per_device_train_batch_size: Optional[int] = field(default=1) 
    gradient_accumulation_steps: Optional[int] = field(default=16)
    learning_rate: Optional[float] = field(default=1e-5)
    num_train_epochs: Optional[int] = field(default=2, metadata={"help": "The number of training epochs for the reward model."})
    optim: Optional[str] = field(default="adamw_hf",  metadata={"help": "The optimizer to use."})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "The lr scheduler"},)
    max_length: Optional[int] = field(default=1024) 
    gradient_checkpointing: Optional[bool] = field(default=True)
    bf16: Optional[bool] = field(default=True)
    attn_implementation: Optional[str] = field(default="flash_attention_2")
    # data
    dataset: Optional[str] = field(default='llm-blender/Unified-Feedback')
    dataset_mode: Optional[str] = field(default='', metadata={"help": "use from '', '40k', and '400k' for the paper's experiments"},)
    # lora
    use_lora: Optional[bool] = field(default=True)
    lora_target_modules: Optional[List[str]] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    lora_r: Optional[int] = field(default=32)
    lora_alpha: Optional[int] = field(default=64)
    lora_dropout: Optional[float] = field(default=0.05)
    # eval
    per_device_eval_batch_size: Optional[int] = field(default=1)
    evaluation_strategy: Optional[str] = field(default="steps")
    eval_steps: Optional[int] = field(default=100)
    # model and loss
    base_model: Optional[str] =  field(default="google/gemma-2b-it")
    loss_type: Optional[str] = field(default='bt', metadata={'help': "use 'bt', 'margin', 'labelsmooth', and 'pos_reg'."})
    weight_ratio: Optional[float] = field(default=1, metadata={'help': 'the ratio for label smooth or posreg'})
    freeze_pretrained: Optional[bool] = field(default=False)
    # log
    report_to: Optional[str] = field(default='none', metadata={'help': "use 'none', 'wandb'. "})
    log_dir: Optional[str] = field(default='./reward_models_train')
    wandb_name: Optional[str] = field(default="test",)
    save_strategy: Optional[str] = field(default="epoch")
    save_steps: Optional[int] = field(default=1000)
    debug: Optional[bool] = field(default=False, metadata={'help': 'if debug=True, only train with 100 samples'})
    
    bt_dataset: Optional[str] = field(default="pharaouk/ultrafeedback-binarized-preferences-cleaned")
    reg_dataset: Optional[str] = field(default="openbmb/UltraFeedback")


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
model_name_split = script_args.base_model.split("/")[-1]
if script_args.use_lora:
    output_name = f"{script_args.log_dir}/{model_name_split}_{script_args.wandb_name}_len{script_args.max_length}_lora{script_args.lora_r}_{script_args.learning_rate}_data{script_args.dataset.split('/')[-1]}"
else:
    output_name = f"{script_args.log_dir}/{model_name_split}_{script_args.wandb_name}_len{script_args.max_length}_fulltrain_{script_args.learning_rate}_data{script_args.dataset.split('/')[-1]}"

device = Accelerator().local_process_index 

training_args = TrainingArguments(
    output_dir=os.path.join(output_name, 'logs'),
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    # evaluation_strategy=script_args.evaluation_strategy,
    evaluation_strategy='no',
    eval_steps=script_args.eval_steps,
    save_strategy=script_args.save_strategy,
    save_steps=script_args.save_steps,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing, 
    bf16=script_args.bf16,
    logging_strategy="steps",
    logging_steps=10,
    warmup_ratio=0.03,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    run_name=script_args.wandb_name,
    max_grad_norm=5.0,
    report_to=script_args.report_to,
    remove_unused_columns=False,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    ddp_find_unused_parameters=False,
)






from transformers import AutoTokenizer, AutoModelForSequenceClassification
from load_datasets import load_train_eval_dataset, build_dataset_helpsteer, CombinedDataset, build_dataset_UF  # ensure these helpers are defined
from reward_trainer import CombinedDataCollator  # make sure this collator is implemented as described
from reward_trainer import SimpleRewardTrainer
from utils import print_trainable_parameters, compute_metrics, freeze_trainable_parameters
from peft import LoraConfig, TaskType
import torch
import os

# Then, after tokenizer initialization:
tokenizer = AutoTokenizer.from_pretrained(script_args.base_model, use_fast=False)
tokenizer.max_length = script_args.max_length
if tokenizer.pad_token is None:
    if 'Llama' in script_args.base_model:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    else:
        tokenizer.pad_token = tokenizer.eos_token

# --- Load the new datasets ---
from load_datasets import build_dataset_bt, build_dataset_regression, CombinedDataset


bt_train_ds = build_dataset_UF('llm-blender/Unified-Feedback', tokenizer, mode = script_args.dataset_mode)
# reg_train_ds = build_dataset_regression(script_args.reg_dataset, tokenizer, split='train', size=100 if script_args.debug else None)
reg_train_ds = build_dataset_helpsteer('nvidia/HelpSteer2', tokenizer, split='train', size=100 if script_args.debug else None)

# For evaluation you might pick one of them, e.g., use BT evaluation samples:
# bt_eval_ds = build_dataset_bt(script_args.bt_dataset, tokenizer, split='validation')  # adjust split name if needed

combined_train_ds = CombinedDataset(bt_train_ds, reg_train_ds)
eval_dataset = combined_train_ds  # or a separate evaluation strategy



# Retrieve a sample from the combined dataset:
sample = combined_train_ds[0]

# BT sample: contains both chosen and rejected texts.
bt_sample = sample["bt"]
# Regression sample:
reg_sample = sample["reg"]

# Decode the texts:
decoded_chosen = tokenizer.decode(bt_sample["input_ids_chosen"], skip_special_tokens=True)
decoded_rejected = tokenizer.decode(bt_sample["input_ids_rejected"], skip_special_tokens=True)
decoded_reg_text = tokenizer.decode(reg_sample["input_ids_reg"], skip_special_tokens=True)

print(combined_train_ds.length)
# Print the examples:
print("=== BT Sample ===")
print("Chosen:")
print(decoded_chosen)
print("\nRejected:")
print(decoded_rejected)

print("\n=== Regression Sample ===")
print("Decoded Text:")
print(decoded_reg_text)
print("Target Attributes:")
print(reg_sample["target_attributes"])


# --- Model Initialization ---
# Note: num_labels is now 5 (1 for BT and 4 for regression)
model = AutoModelForSequenceClassification.from_pretrained(
    script_args.base_model, num_labels=6, device_map=device, 
    torch_dtype=torch.bfloat16,
    **({"attn_implementation": script_args.attn_implementation} if script_args.attn_implementation else {})
)

# (Keep the rest of the model configuration as before, including freezing if needed.)

model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
print_trainable_parameters(model)

# --- Use the CombinedDataCollator ---
from reward_trainer import CombinedDataCollator
data_collator = CombinedDataCollator(tokenizer=tokenizer, max_length=script_args.max_length)

# --- Trainer Parameters ---
trainer_params = {
    "model": model,
    "args": training_args,
    "tokenizer": tokenizer,
    "train_dataset": combined_train_ds,
    "eval_dataset": eval_dataset,
    "compute_metrics": compute_metrics,  # adjust if needed for your new evaluation
    "data_collator": data_collator,
    "loss_type": script_args.loss_type,
    "weight_ratio": script_args.weight_ratio,
}
# (Include PEFT config if you still wish to use LoRA.)
if script_args.use_lora:
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        target_modules=script_args.lora_target_modules,
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
    )
    trainer_params["peft_config"] = peft_config

trainer = SimpleRewardTrainer(**trainer_params)
print_trainable_parameters(trainer.model)

print('training start')
trainer.train()

