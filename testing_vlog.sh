#!/usr/bin/env bash

# Envs
# source activate pytorch-0.4.0

# Pythonpath
PYTHONPATH=.

python main.py --root $VLOG \
--resume $1 \
--blocks 2D_2D_2D_2.5D \
--object-head 2D \
--add-background \
--train-set train+val \
--arch orn_two_heads \
--depth 50 \
-t 4 \
-b 16 \
--cuda \
--dataset vlog \
--heads object+context \
-j 4 \
--nb-crops 8 \
--mask-confidence 0.75 \
-e
