lr=1e-5
block_size=2048
# pretrained_model=/data/disk0/Baichuan2-7B-Base
pretrained_model=/data/chinese-llama-2-7b
# dataset_dir=/data/disk0/dataset/wanjuan/OpenDataLab___WanJuan1_dot_0/raw/nlp/CN/Patent-cn/,/data/disk0/dataset/wanjuan/OpenDataLab___WanJuan1_dot_0/raw/nlp/CN/Exam-cn/,/data/disk0/dataset/tigerbot/zh/pretrain_zh/data/
dataset_dir=/data/dataset/nlp-xingyu-dataset/recommed/train_data/
# data_cache=/data/disk1/temp_data_cache_dir/tigerbot_zh_wanjuan_and_baichuan2_7b_${block_size}
data_cache=/data/temp_data_cache_dir/tigerbot_zh_2048
per_device_train_batch_size=2
per_device_eval_batch_size=2
gradient_accumulation_steps=8
output_dir=/data/disk1/output_tigerbot_zh_wanjuan_and_baichuan2_7b_pt_${block_size}

deepspeed_config_file=ds_zero3_1.json

#torchrun --nnodes 2 --nproc_per_node 8 --node_rank 0 --master_addr=10.1.2.179 --master_port=23457 run_clm_pt.py \
# export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
torchrun --nnodes 1 --nproc_per_node 8 run_multi_data_and_auto_model_pt.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${pretrained_model} \
    --dataset_dir ${dataset_dir} \
    --data_cache_dir ${data_cache} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --do_eval \
    --validation_split_percentage 0.003 \
    --evaluation_strategy steps \
    --eval_steps 250 \
    --seed 4 \
    --fp16 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.01 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 2 \
    --save_steps 250 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 100 \
    --block_size ${block_size} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --torch_dtype float16 \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False
