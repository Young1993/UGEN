# for 5-shot gradient_accumulation_steps 20 print_steps 20
# for 10-shot gradient_accumulation_steps 20 print_steps 20

mode=$1
gpu=0
echo "mode: $mode"

CUDA_VISIBLE_DEVICES=$gpu python3 train_qa.py \
  --gradient_accumulation_steps 60 \
  --batch_size 16 \
  --print_steps 60 \
  --mode $mode \
  --lr 3e-5 \
  --max_epoch 100 \
  --visible_gpu 0