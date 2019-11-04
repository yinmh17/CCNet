#!/usr/bin/env bash

CS_PATH=$1
PRE_PATH=$2
CONFIG=$3
LIST1='./dataset/list/cityscapes/test1.lst'
LIST2='./dataset/list/cityscapes/test2.lst'
LIST3='./dataset/list/cityscapes/test3.lst'
LIST4='./dataset/list/cityscapes/test4.lst'

python test_multi.py --data-dir ${CS_PATH} --data-list ${LIST1} --restore-from ${PRE_PATH} --gpu 0 --whole --recurrence 1 --config ${CONFIG} &
python test_multi.py --data-dir ${CS_PATH} --data-list ${LIST2} --restore-from ${PRE_PATH} --gpu 1 --whole --recurrence 1 --config ${CONFIG} &
python test_multi.py --data-dir ${CS_PATH} --data-list ${LIST3} --restore-from ${PRE_PATH} --gpu 2 --whole --recurrence 1 --config ${CONFIG} &
python test_multi.py --data-dir ${CS_PATH} --data-list ${LIST4} --restore-from ${PRE_PATH} --gpu 3 --whole --recurrence 1 --config ${CONFIG} &
