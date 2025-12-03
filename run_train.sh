export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH="/home/zhuomin/leo/spad-fast":$PYTHONPATH

# nohup python scripts/distill.py \
#   --data_root training \
#   --config configs/spad_two_views.yaml \
#   --batch_size 8 \
#   --accumulate_steps 8 \
#   --num_views 2 \
#   --image_size 256 \
#   --num_epochs 30 \
#   --max_steps 2000000 \
#   --fp16 \
#   --output_dir logs/spad_dubug_dataloader \
#   --log_every 10 \
#   --ckpt_every 350 \
#   --save_last \
#   > train.log 2>&1 &

# python scripts/distill.py \
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
#   --save_last



nohup python scripts/distill.py \
  --config configs/spad_two_views.yaml \
  --teacher_ckpt data/checkpoints/spad_two_views.ckpt \
  --data_root training \
  --batch_size 2 \
  --accumulate_steps 16 \
  --num_views 2 \
  --image_size 256 \
  --train_timesteps 1000 \
  --num_sampling_steps 4 \
  --learning_rate 1e-4 \
  --max_steps 3000 \
  --log_every 10 \
  --ckpt_every 1000 \
  --output_dir logs/spad_lcm_lora \
  > distill.log 2>&1 &