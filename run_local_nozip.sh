#!/bin/bash
uname -a
date

CS_PATH=$1
PRE_PATH=$2
CONFIG=$3
R=1
STEPS=60000
GPU_IDS=0,1,2,3
START=0
OHEM=1 #set to 1 for reducing the performance gap between val and test set.

#variable ${LOCAL_OUTPUT} dir can save data of you job, after exec it will be upload to hadoop_out path 
python train.py --data-dir ${CS_PATH} --random-mirror --random-scale --restore-from ${PRE_PATH} --config ${CONFIG} --gpu ${GPU_IDS} --start-iters ${START} --ohem ${OHEM} --ohem-thres 0.7 --ohem-keep 100000 --use-zip False
#python evaluate.py --data-dir ${CS_PATH} --restore-from snapshots/CS_scenes_${STEPS}.pth --gpu 0 --recurrence ${R} --config ${CONFIG}
python evaluate_all.py --data-dir ${CS_PATH} --restore-from snapshots/CS_scenes_${STEPS}.pth --gpu 0 --recurrence ${R} --config ${CONFIG} --use-zip False
