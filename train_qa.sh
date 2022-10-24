# for 5-shot gradient_accumulation_steps 20 print_steps 20
# for 10-shot gradient_accumulation_steps 20 print_steps 20

mode=$1
gpu=0
dataset_name=$2
dataset_dir=$3
echo "mode: $mode"

CUDA_VISIBLE_DEVICES=$gpu python3 train_qa.py \
  --gradient_accumulation_steps 60 \
  --batch_size 16 \
  --print_steps 60 \
  --mode $mode \
  --lr 3e-5 \
  --max_epoch 100 \
  --dataset_name $dataset_name \
  --dataset_dir $3\
  --visible_gpu 0