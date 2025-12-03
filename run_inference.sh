export PYTHONPATH="/home/zhuomin/leo/spad-fast":$PYTHONPATH

python scripts/fast_inference.py \
  --captions "Yellow Toyota Celica sports car." \
  --model spad_two_views \
  --ema_ckpt logs/spad_lcm_lora/last.pt \
  --cfg_scale 7.5 \
  --num_sampling_steps 4 \
  --train_timesteps 1000 \
  --batch_size 1 \
  --total_views 8
