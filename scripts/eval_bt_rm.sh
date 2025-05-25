
gpu='0,1,2,3'
port=8991
max_length=1024
per_device_eval_batch_size=4
base_model="/workdir/mistral-smorm/save_reward_models/Mistral-7B-Instruct-v0.2_BT_RM_len3000_fulltrain_5e-06_dataSkywork-Reward-Preference-80K-v0.2/logs/checkpoint-828"
## If not use lora, make peft_name=''
# peft_name='../save_reward_models/gemma-2b-it_BT_RM_seed2_len1024_lora32_1e-05_dataUnified-Feedback/logs/checkpoint-3536'
log_dir='./eval_BT'
save_all_data=False
freeze_pretrained=False # for freeze pretrained feature baseline

cd ../rm_eval
for task in  'unified' 'hhh'  'mtbench' 
do 
    CUDA_VISIBLE_DEVICES=${gpu} accelerate launch --main_process_port ${port} eval.py \
                                        --base_model ${base_model} \
                                        --per_device_eval_batch_size ${per_device_eval_batch_size} \
                                        --save_all_data ${save_all_data} --freeze_pretrained ${freeze_pretrained} \
                                        --task ${task}  --max_length ${max_length} --log_dir ${log_dir}
done



