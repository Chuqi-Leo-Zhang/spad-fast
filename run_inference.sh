export PYTHONPATH="/home/zhuomin/leo/spad-fast":$PYTHONPATH

# python scripts/fast_inference.py \
#   --captions "Yellow Toyota Celica sports car." \
#   --model spad_two_views \
#   --ema_ckpt logs/spad_lcm_lora/last.pt \
#   --cfg_scale 7.5 \
#   --num_sampling_steps 4 \
#   --train_timesteps 1000 \
#   --batch_size 1 \
#   --total_views 8


# python scripts/fast_inference.py \
#   --config configs/spad_two_views.yaml \
#   --teacher_ckpt data/checkpoints/spad_two_views.ckpt \
#   --ema_lora_ckpt logs/spad_lcm_lora/last.pt \
#   --captions "Yellow Toyota Celica sports car." \
#   --cfg_scale 7.5 \
#   --batch_size 1 \
#   --total_views 8 \
#   --train_timesteps 1000 \
#   --num_sampling_steps 4 \
#   --lora_rank 16 \
#   --lora_alpha 16


python scripts/fast_inference.py \
  --config configs/spad_two_views.yaml \
  --teacher_ckpt data/checkpoints/spad_two_views.ckpt \
  --ema_lora_ckpt logs/spad_lcm_lora/step_0002000.pt \
  --captions "Yellow Toyota Celica sports car." \
  --ddim_steps 4 \
  --cfg_scale 7.5 \
  --batch_size 1 \
  --total_views 8
