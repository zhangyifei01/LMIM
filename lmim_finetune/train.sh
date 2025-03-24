# Set the path to save checkpoints
OUTPUT_DIR='output/'
# path to imagenet-1k set
DATA_PATH='/path/to/finetune_data'
# path to pretrain model
MODEL_PATH='../lmim_pretrain/output_dir/checkpoint-19.pth'

# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --master_port 10041 run_class_finetuning.py \
    --model simmim_vit_small_patch4_32x128 \
    --data_path /data1/ImageData/Union14M-L-lmdb/train/train_challenging /data1/ImageData/Union14M-L-lmdb/train/train_easy /data1/ImageData/Union14M-L-lmdb/train/train_hard /data1/ImageData/Union14M-L-lmdb/train/train_medium /data1/ImageData/Union14M-L-lmdb/train/train_normal \
    --eval_data_path /data1/ImageData/Union14M-L-lmdb/val/val_annos \
    --finetune ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 64 \
    --opt adamw \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --data_set image_lmdb \
    --nb_classes 97 \
    --smoothing 0. \
    --max_len 25 \
    --epochs 10 \
    --warmup_epochs 1 \
    --drop 0.1 \
    --attn_drop_rate 0.1 \
    --drop_path 0.1 \
    --dist_eval \
    --lr 1e-4 \
    --num_samples 1 \
    --fixed_encoder_layers 0 \
    --decoder_name tf_decoder \
    --use_abi_aug \
    --num_view 2 \

# Set the path to save checkpoints
OUTPUT_DIR='output/eval'
DATA_PATH='/data1/ImageData/Union14M/Union14M-L/Union14M-Benchmarks/lmdb_format/general'
# path to finetune model
MODEL_PATH='./output/checkpoint-9.pth'
opt_nproc_per_node=1

# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=$opt_nproc_per_node --master_port 10040 run_class_finetuning.py \
    --model simmim_vit_small_patch4_32x128 \
    --data_path ${DATA_PATH} \
    --eval_data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 128 \
    --opt adamw \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --data_set image_lmdb \
    --nb_classes 97 \
    --smoothing 0. \
    --max_len 25 \
    --resume ${MODEL_PATH} \
    --eval \
    --epochs 20 \
    --warmup_epochs 2 \
    --drop 0.1 \
    --attn_drop_rate 0.1 \
    --dist_eval \
    --num_samples 1000000 \
    --fixed_encoder_layers 0 \
    --decoder_name tf_decoder \
    --beam_width 0 
