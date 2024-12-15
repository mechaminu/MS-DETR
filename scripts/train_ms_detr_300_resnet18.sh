coco_path=/workspace/coco2017
EXP_DIR=/workspace/ms-detr-checkpoints/ms_detr_300_resnet18
num_gpus=3

mkdir -p $EXP_DIR

GPUS_PER_NODE=$num_gpus ./tools/run_dist_launch.sh $num_gpus python -u main.py \
   --output_dir $EXP_DIR \
   --with_box_refine \
   --two_stage \
   --dim_feedforward 2048 \
   --epochs 12 \
   --lr_drop 11 \
   --coco_path=$coco_path \
   --num_queries 300 \
   --use_ms_detr \
   --use_aux_ffn \
   --topk_eval 100 \
   --backbone resnet18 \
   --resume $EXP_DIR/checkpoint.pth
   > $EXP_DIR/train.log
