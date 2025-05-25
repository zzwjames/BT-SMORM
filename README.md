
## Usage 
First set the environment variable.
```
export HF_HOME='your HF token'
```
Then install the environment. Note that we found error in transformers==4.46.1, please use early versions.
```
pip install -r requirements.txt
```

### Training and evaluation of reward models
Then, go to the `scripts' folder and train the reward model with the default hyperparameters
```
cd scripts
sh train_bt_rm_full.sh
sh train_bt_rm_lora.sh
sh train_grm_full.sh
sh train_grm_lora.sh
```

Evaluating trained models on 'llm-blender/Unified-Feedback', 'HuggingFaceH4/hhh_alignment', 'lmsys/mt_bench_human_judgments':
```
sh eval_bt_rm.sh
sh eval_grm_rm.sh
```

### RLHF

#### Data Generation

```
cd scripts/rlhf
sh data_generation4rlhf.sh
```

#### BoN
Go to the `scripts/rlhf/bon` folder and run the scripts. For more information about each step, please refer to `rlhf/bon/README.md` .

**Note: please set the path to your dataset and reward model in the corresponding shells.**
```
cd scripts/rlhf/bon
sh step1_train_proxy_reward_model_baseline.sh
sh step1_train_proxy_reward_model_grm.sh
sh step2_generate_samples.sh
sh step3_obtain_proxy_score.sh
sh step4_choose_best_of_n.sh
sh step5_obtain_bon_gold_score.sh
sh step6_collect.sh
```

#### PPO
Go to the `scripts/rlhf/ppo' folder and train the gemma-2b-it model with the default parameters.

**Note: please set the path to your reward model in the corresponding shells.**
```
cd scripts/rlhf/ppo
sh train_ppo.sh
sh train_ppo.grm.sh
sh train_ppo_ensemble.sh
```


