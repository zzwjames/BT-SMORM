from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from accelerate import Accelerator
import numpy as np
import torch
import torch.nn as nn
from base_trainer import RewardTrainer
from transformers.utils import PaddingStrategy
from transformers import AutoTokenizer



@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged_features = []
        margins = []
        for feature in features:
            merged_features.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )
            merged_features.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )
            if 'margin' in feature.keys():
                margins.append(feature['margin'])
        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "return_loss": True,
            "margin": margins,
        }
        return batch

class CombinedDataCollator:
    def __init__(self, tokenizer, max_length=None, pad_to_multiple_of=None, padding="max_length", return_tensors="pt"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.padding = padding  # use "max_length" to force fixed-length padding
        self.return_tensors = return_tensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Process skywork examples
        skywork_features = []
        margins = []  # if applicable from skywork
        for f in features:
            sky = f["skywork"]
            skywork_features.append({
                "input_ids": sky["input_ids_chosen"],
                "attention_mask": sky["attention_mask_chosen"],
                "sample_type": 0,  # mark as skywork
            })
            skywork_features.append({
                "input_ids": sky["input_ids_rejected"],
                "attention_mask": sky["attention_mask_rejected"],
                "sample_type": 0,
            })
            if 'margin' in sky:
                margins.append(sky["margin"])
        
        # Process helpsteer examples
        helpsteer_features = []
        target_attributes = []
        for f in features:
            hs = f["helpsteer"]
            helpsteer_features.append({
                "input_ids": hs["input_ids_helpsteer"],
                "attention_mask": hs["attention_mask_helpsteer"],
                "sample_type": 1,
            })
            target_attributes.append(hs["target_attributes"])
        
        # Use fixed-length padding for both batches by setting padding="max_length" and max_length.
        skywork_batch = self.tokenizer.pad(
            [ {k: v for k, v in feat.items() if k in ["input_ids", "attention_mask"]} for feat in skywork_features ],
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        skywork_batch["sample_type"] = torch.tensor([feat["sample_type"] for feat in skywork_features])
        
        helpsteer_batch = self.tokenizer.pad(
            [ {k: v for k, v in feat.items() if k in ["input_ids", "attention_mask"]} for feat in helpsteer_features ],
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        helpsteer_batch["sample_type"] = torch.tensor([feat["sample_type"] for feat in helpsteer_features])
        helpsteer_batch["target_attributes"] = torch.stack(target_attributes)

        # Concatenate batches along the batch dimension. With fixed-length padding, these tensors should have matching sizes.
        batch = {
            "input_ids": torch.cat([skywork_batch["input_ids"], helpsteer_batch["input_ids"]], dim=0),
            "attention_mask": torch.cat([skywork_batch["attention_mask"], helpsteer_batch["attention_mask"]], dim=0),
            "sample_type": torch.cat([skywork_batch["sample_type"], helpsteer_batch["sample_type"]], dim=0),
        }
        if margins:
            batch["margin"] = margins
        
        # Create a placeholder for target attributes for all items, filling skywork positions with zeros.
        help_count = helpsteer_batch["input_ids"].size(0)
        total = batch["input_ids"].size(0)
        target_tensor = torch.zeros(total, 5)
        # Assuming helpsteer samples are at the end of the batch:
        target_tensor[-help_count:] = helpsteer_batch["target_attributes"]
        batch["target_attributes"] = target_tensor
        return batch


class CombinedDataCollator:
    def __init__(self, tokenizer, max_length=None, pad_to_multiple_of=None, padding="max_length", return_tensors="pt"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.padding = padding  # use "max_length" for fixed-length padding
        self.return_tensors = return_tensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Process BT samples (paired chosen and rejected)
        bt_features = []
        for f in features:
            bt = f["bt"]
            bt_features.append({
                "input_ids": bt["input_ids_chosen"],
                "attention_mask": bt["attention_mask_chosen"],
                "sample_type": 0,
            })
            bt_features.append({
                "input_ids": bt["input_ids_rejected"],
                "attention_mask": bt["attention_mask_rejected"],
                "sample_type": 0,
            })
        
        # Process Regression samples
        reg_features = []
        target_attributes_list = []
        for f in features:
            reg = f["reg"]
            reg_features.append({
                "input_ids": reg["input_ids_reg"],
                "attention_mask": reg["attention_mask_reg"],
                "sample_type": 1,
            })
            target_attributes_list.append(reg["target_attributes"])
        
        bt_batch = self.tokenizer.pad(
            [{k: v for k, v in feat.items() if k in ["input_ids", "attention_mask"]}
             for feat in bt_features],
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        bt_batch["sample_type"] = torch.tensor([feat["sample_type"] for feat in bt_features])
        
        reg_batch = self.tokenizer.pad(
            [{k: v for k, v in feat.items() if k in ["input_ids", "attention_mask"]}
             for feat in reg_features],
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        reg_batch["sample_type"] = torch.tensor([feat["sample_type"] for feat in reg_features])
        reg_batch["target_attributes"] = torch.stack(target_attributes_list)
        
        # Concatenate the two batches along the batch dimension.
        batch = {
            "input_ids": torch.cat([bt_batch["input_ids"], reg_batch["input_ids"]], dim=0),
            "attention_mask": torch.cat([bt_batch["attention_mask"], reg_batch["attention_mask"]], dim=0),
            "sample_type": torch.cat([bt_batch["sample_type"], reg_batch["sample_type"]], dim=0),
        }
        # For regression, create a placeholder for target attributes: zeros for BT samples.
        reg_count = reg_batch["input_ids"].size(0)
        total = batch["input_ids"].size(0)
        target_tensor = torch.zeros(total, 5)  # 4 regression targets
        # Place the regression targets at the end of the batch.
        target_tensor[-reg_count:] = reg_batch["target_attributes"]
        batch["target_attributes"] = target_tensor
        return batch




class SimpleRewardTrainer(RewardTrainer):
    def __init__(self, **kwargs):
        self.loss_type = kwargs.pop('loss_type', 'bt')
        self.weight_ratio = kwargs.pop('weight_ratio', 0.1)
        super(SimpleRewardTrainer, self).__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])[0]
        sample_types = inputs["sample_type"]  # tensor of 0s and 1s

        # print('outputs', outputs)
        loss = 0.0
        loss_dict = {}

        # --- BT loss for BT samples (sample_type==0) ---
        bt_indices = (sample_types == 0).nonzero(as_tuple=True)[0]
        if bt_indices.numel() > 0:
            bt_outputs = outputs[bt_indices]
            # Pair the outputs: even index is chosen, odd index is rejected.
            chosen = bt_outputs[0::2, 0]  # first logit
            # print('chosen', chosen)
            rejected = bt_outputs[1::2, 0]
            # print('rejected', rejected)
            bt_loss = - torch.nn.functional.logsigmoid(chosen - rejected).mean()
            loss += bt_loss
            loss_dict["bt_loss"] = bt_loss

        # --- Regression loss for regression samples (sample_type==1) ---
        reg_indices = (sample_types == 1).nonzero(as_tuple=True)[0]
        if reg_indices.numel() > 0:
            reg_outputs = outputs[reg_indices]
            # Use logits 1 to 4 for regression.
            preds = reg_outputs[:, 1:]
            # print('preds', preds)
            targets = inputs["target_attributes"][reg_indices]
            # print('targets', targets)
            reg_loss = torch.nn.functional.mse_loss(preds, targets)
            loss += reg_loss*self.weight_ratio
            loss_dict["reg_loss"] = reg_loss

        # print('bt_loss', bt_loss)
        # print('loss', loss)

        if return_outputs:
            return loss, loss_dict
        return loss


