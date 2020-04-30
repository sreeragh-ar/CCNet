#!/bin/bash
uname -a
#date
#env
date

CS_PATH=/media/sdh/vishal/cityscapes
MODEL=ccnet
LR=1e-2
WD=5e-4
BS=4
STEPS=$1
INPUT_SIZE=500,500
OHEM=1
GPU_IDS=1,2

#variable ${LOCAL_OUTPUT} dir can save data of you job, after exec it will be upload to hadoop_out path 
CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch --nproc_per_node=2 train.py --data-dir ${CS_PATH} --model ${MODEL} --random-mirror --random-scale --restore-from ./dataset/resnet101-imagenet.pth --input-size ${INPUT_SIZE} --gpu ${GPU_IDS} --learning-rate ${LR}  --weight-decay ${WD} --batch-size ${BS} --num-steps ${STEPS} --ohem ${OHEM} --recurrence 2 --save-pred-every 500 &&
CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch --nproc_per_node=2 evaluate.py --data-dir ${CS_PATH} --model ${MODEL} --input-size ${INPUT_SIZE} --batch-size 4 --restore-from snapshots/CS_scenes_${STEPS}.pth --gpu 2