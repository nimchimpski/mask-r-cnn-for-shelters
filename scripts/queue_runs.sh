#!/bin/bash
mkdir -p logs checkpoints
conda run -n camp_maskrcnn python scripts/train_maskrcnn.py --run 5b --backbone-layers 3 --aug full --min-area 25 2>&1 | tee logs/5b_stdout.log
conda run -n camp_maskrcnn python scripts/train_maskrcnn.py --run 5a --backbone-layers 3 2>&1 | tee logs/5a_stdout.log