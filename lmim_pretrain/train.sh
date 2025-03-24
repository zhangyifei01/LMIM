TEACHER_MODEL_PATH='../vit_small_checkpoint-19.pth'

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
        --nproc_per_node=8 main_pretrain.py \
        --teacher_weight ${TEACHER_MODEL_PATH} \
        --batch_size 64 \
        --model mae_vit_small_patch4 \
        --mask_ratio 0.80 \
        --epochs 20 \
        --warmup_epochs 1 \
        --norm_pix_loss \
        --blr 1.5e-4 \
        --weight_decay 0.05 \
        --data_path /data1/ImageData/Union14M-U/book32_lmdb /data1/ImageData/Union14M-U/cc_lmdb /data1/ImageData/Union14M-U/openvino_lmdb
        

