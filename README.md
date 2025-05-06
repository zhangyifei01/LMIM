## Linguistics-aware Masked Image Modeling for Self-supervised Scene Text Recognition

<p align="center">
  <img src=figs/fig1.png width="500">
</p>

### Unsupervised Pre-training


Pre-trained [vit_small_checkpoint-19.pth](https://drive.google.com/file/d/1F_wK7iAYzyz-T7-4D_dNQo1ORZ2Kesyw/view) from [MAERec](https://github.com/Mountchicken/Union14M).


```
cd ./lmim_pretrain

TEACHER_MODEL_PATH='../vit_small_checkpoint-19.pth'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
        --nproc_per_node=8 main_pretrain.py \
        --teacher_weight ${TEACHER_MODEL_PATH} \
        --batch_size 64 \
        --model mae_vit_small_patch4 \
        --mask_ratio 0.80 \
        --epochs 10 \
        --warmup_epochs 1 \
        --norm_pix_loss \
        --blr 1.5e-4 \
        --weight_decay 0.05 \
        --data_path [your Union14M-U lmdb folder]
 
```

### Downstream Recognition

STR
```
cd ../lmim_finetune/

OUTPUT_DIR='output/'
DATA_PATH='/path/to/finetune_data'
MODEL_PATH='../lmim_pretrain/output_dir/checkpoint-9.pth'

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --master_port 10041 run_class_finetuning.py \
    --model simmim_vit_small_patch4_32x128 \
    --data_path [your Union14M-L-lmdb/train folder, e.g. Union14M-L-lmdb/train/train_challenging] \
    --eval_data_path [your Union14M-L-lmdb/val folder]  \
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
```

### Evaluation
```
cd ../lmim_finetune/

OUTPUT_DIR='output/eval'
DATA_PATH=[your Union14M-Benchmarks folder, e.g. Union14M/Union14M-L/Union14M-Benchmarks/lmdb_format/general]
MODEL_PATH='./output/checkpoint-9.pth'
opt_nproc_per_node=1

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
```

### Setting

The recognizer uses a 12-layer ViT-Small as the encoder and a 6-layer Transformer as the decoder. For the English dataset, the total number of categories is 97, the maximum sequence length is set to 25, and the number of trainable parameters is 35.8M. For the Chinese dataset, the total number of categories is 7937, the maximum sequence length is set to 40, and the number of trainable parameters is 43.8M.

### Dataset

* English pre-train： [Union14M](https://github.com/Mountchicken/Union14M)

* Chinese pre-train: [Unlabeled Chinese Text Image 11M (UCTI-11M)](https://pan.baidu.com/s/1ikQWhwagpP4lScwUVehpbw) (code: pbpn)

* [Six common benchmarks: IIIT5k et.al.](https://github.com/ku21fan/STR-Fewer-Labels/blob/main/data.md) & [Verification](https://github.com/Xiaomeng-Yang/STR_benchmark_cleansed)

* [Chinese benchmark](https://github.com/FudanVI/benchmarking-chinese-text-recognition)



### Citation
```
@inproceedings{zhang2025lmim,
  author  = {Zhang, Yifei and Liu, Chang and Wei, Jin and Yang, Xiaomeng and Zhou, Yu and Ma, Can and Ji, Xiangyang},
  title   = {Linguistics-aware masked image modeling for self-supervised scene text recognition},
  booktitle = {Proceedings of the IEEE/CVF Conferences on Computer Vision and Pattern Recognition (CVPR)},
  pages   = {},
  year = {2025}
}
```
