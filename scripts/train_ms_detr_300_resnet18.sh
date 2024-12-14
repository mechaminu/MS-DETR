coco_path=/workspace/coco2017
num_gpus=3

EXP_DIR=/workspace/ms-detr-checkpoints/ms_detr_300_resnet18

mkdir -p $EXP_DIR

GPUS_PER_NODE=$num_gpus ./tools/run_dist_launch.sh $num_gpus python -u main.py \
   --output_dir /workspace/ms-detr-checkpoints/ms_detr_300_resnet18 \
   --with_box_refine \
   --two_stage \
   --dim_feedforward 2048 \
   --epochs 12 \
   --lr_drop 11 \
   --coco_path=/workspace/coco2017 \
   --num_queries 300 \
   --use_ms_detr \
   --use_aux_ffn \
   --topk_eval 100 \
   --backbone resnet18 \
   --resume /workspace/ms-detr-checkpoints/ms_detr_300_resnet18/checkpoint.pth
   > $EXP_DIR/train.log
