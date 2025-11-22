export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH="/home/zhuomin/leo/spad-fast":$PYTHONPATH

# nohup python scripts/distill.py \
#   --data_root training_data/training \
#   --config configs/spad_two_views.yaml \
#   --batch_size 8 \
#   --num_views 2 \
#   --image_size 256 \
#   --num_epochs 100 \
#   --max_steps 2000000 \
#   --fp16 \
#   --output_dir logs/spad_debug \
#   --log_every 100 \
#   --ckpt_every 10000 \
#   --save_last \
#   > train.log 2>&1 &

python scripts/distill.py \
  --data_root training_data/training \
  --config configs/spad_two_views.yaml \
  --batch_size 8 \
  --num_views 2 \
  --image_size 256 \
  --num_epochs 100 \
  --max_steps 2000000 \
  --fp16 \
  --output_dir logs/spad_debug \
  --log_every 100 \
  --ckpt_every 10000 \
  --save_last

