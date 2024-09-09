PRE_SEQ_LEN=128
LR=2e-2
NUM_GPUS=1
export CUDA_VISIBLE_DEVICES=1

torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS /mnt/data/cty/works/chatglm2-6b/ptuning/main.py \
    --do_train \
    --train_file /mnt/data/cty/works/chatglm2-6b/preprocess/build_datasets/detection/dapt-2020-new/dapt-2020_detection_packet_train.json \
    --validation_file /mnt/data/cty/works/chatglm2-6b/preprocess/build_datasets/detection/dapt-2020-new/dapt-2020_detection_packet_test.json \
    --preprocessing_num_workers 10 \
    --prompt_column instruction \
    --response_column output \
    --overwrite_cache \
    --cache_dir /mnt/data/cty/cache \
    --model_name_or_path /mnt/data/cty/models/chatglm2/chatglm2-6b \
    --output_dir /mnt/data/cty/models/chatglm2/td/dapt-2020-new-chatglm2-6b-ft-packet \
    --overwrite_output_dir \
    --max_source_length 1024 \
    --max_target_length 32 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 20000 \
    --logging_steps 10 \
    --save_steps 4000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
#    --quantization_bit 4

