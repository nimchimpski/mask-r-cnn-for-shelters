#!/bin/bash
mkdir -p logs checkpoints
conda run -n camp_maskrcnn python scripts/train_maskrcnn.py --run 5b --backbone-layers 3 --aug full --min-area 25 2>&1 | tee logs/5b_stdout.log
conda run -n camp_maskrcnn python scripts/train_maskrcnn.py --run 5a --backbone-layers 3 2>&1 | tee logs/5a_stdout.log

# learning curve (STATUS next steps #3): same config as 5b, 25%/50% of train tiles;
# 5b itself (full train set) is the 100% point, not rerun here
conda run -n camp_maskrcnn python scripts/train_maskrcnn.py --run 5b_lc25 --backbone-layers 3 --aug full --min-area 25 --train-frac 0.25 2>&1 | tee logs/5b_lc25_stdout.log
conda run -n camp_maskrcnn python scripts/train_maskrcnn.py --run 5b_lc50 --backbone-layers 3 --aug full --min-area 25 --train-frac 0.5 2>&1 | tee logs/5b_lc50_stdout.log