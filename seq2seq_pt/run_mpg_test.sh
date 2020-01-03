#!/bin/bash

set -x

DATAHOME=/path/to/MPG/math_problem_generation/dolphint1_eval
EXEHOME=/path/to/MPG/seq2seq_pt

SAVEPATH=${DATAHOME}/models/magnet

mkdir -p ${SAVEPATH}

cd ${EXEHOME}

python translate.py \
       -model ${SAVEPATH}/model_dev_metric_0.1241_e14.pt \
       -src ${DATAHOME}/dev/dev.equ \
       -batch_size 1 -beam_size 3 \
       -gpu 0 \
       -output /tmp/mpg.dev.s2s.batch1.beam3.out
