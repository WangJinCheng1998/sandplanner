#!/bin/bash
# 切换到脚本所在目录(仓库根) / cd to the script's directory (repo root)
cd "$(dirname "$0")"

# 数据集与统计文件路径,可用环境变量覆盖 / Dataset and stats paths; override via env vars
DATASET_ROOT="${DATASET_ROOT:-dataset}"
STATS_JSON="${STATS_JSON:-data/outputs/trajectory_stats_mg2_mg50_2d.json}"

python3 -u sand_planner/training/train_bspline_ddpm.py \
  --dataset_root "$DATASET_ROOT" \
  --batch_size 256 \
  --sequence_length 4 \
  --min_gap 2 \
  --max_gap 42 \
  --samples_per_epoch 19200 \
  --epochs 400\
  --image_height 168 \
  --image_width 224 \
  --downscale_depth \
  --fix_first_cp_zero \
  --normalize_targets \
  --num_workers 8 \
  --norm_method percentile \
  --norm_margin 0.0 \
  --norm_clamp \
  --learning_rate 2e-4 \
  --use_lr_scheduler \
  --lr_scheduler_type cosine \
  --lr_warmup_epochs 10 \
  --lr_min_factor 0.01 \
  --static_sequence_prob 0.1 \
  --multi_frame_fusion concat \
  --transformer_layers 4 \
  --transformer_heads 4 \
  --stats_json_path "$STATS_JSON" \
  --no_type_embed \
  --use_initial_turn \
  --wandb_project sand-planner \
  --prediction_mode control_points \
  --trajectory_interpolation bspline \
  --num_control_points 12 \
 # --max_arc_length 4
