# 在 v100 机器上使用已训练好的 qwen 模型报 output tensor must have the same type as input tensor 这个错误
# 可能是配置文件问题（比如/data/output_model/multi_data_and_qwen_7b_pt_4096/checkpoint-8750/config.json）
# v100 目前已知问题：
#   v100 不支持 bf16，需要设置为 false
#   v100 不支持 use_flash_attn，需要设置为 "auto" # 非必须
# 如仍有问题，可以参考 base 模型修改
# baichuan 模型切换 eval 有问题，先忽略 eval 阶段错误

lr=1e-5

max_seq_length=4096
# pretrained_model=/data/output_model/multi_data_and_baichuan_7b_pt_qiji_4096
pretrained_model=/data/output_model_sft/qwen-pt_6e-sft_sample_2000_6e-l4096-all_and_kg_album-2023_10_20-half
# dataset_dir=/data/dataset/nlp-xingyu-dataset/recommed/sft-data/
dataset_dir=/data/yun.tao/
data_cache_dir=/data/temp_data_cache_dir_sft/Test-Qwen-7B-${max_seq_length}
per_device_train_batch_size=1
per_device_eval_batch_size=1
gradient_accumulation_steps=8
output_dir=/data/output_model_sft/test-qwen-sft-yzy-new-v3
# 如果需要指定 eval 文件加 --validation_file ${validation_file}，默认用 validation_split_percentage 参数自动切分
# validation_file=/data/dataset-sft/test/kg_simple_question.jsonl

deepspeed_config_file=ds_zero3_1.json

# 1
# deepspeed --include localhost:0,1,2,3 run_auto_model_sft.py \

# 2
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nnodes 1 --nproc_per_node 8 run_auto_model_sft.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${pretrained_model} \
    --dataset_dir ${dataset_dir} \
    --data_cache_dir ${data_cache_dir} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --seed 4 \
    --evaluation_strategy "no" \
    --fp16 \
    --num_train_epochs 2 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.01 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 5 \
    --save_strategy steps \
    --save_total_limit 2 \
    --save_steps 30 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 32 \
    --max_seq_length ${max_seq_length} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --torch_dtype float16 \
    --gradient_checkpointing \
    --use_cache True \
    --ddp_find_unused_parameters False
    # --validation_split_percentage 0.01 \
    # --do_eval \

