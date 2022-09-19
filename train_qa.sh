# for 5-shot gradient_accumulation_steps 20 print_steps 20
# for 10-shot gradient_accumulation_steps 20 print_steps 20
#CUDA_VISIBLE_DEVICES=0

mode=$1
echo "mode: $mode"

python3 train_qa.py \
  --gradient_accumulation_steps 60 \
  --batch_size 32 \
  --print_steps 60 \
  --mode $mode \
  --lr 3e-5 \
  --max_epoch 60 \
  --visible_gpu 0