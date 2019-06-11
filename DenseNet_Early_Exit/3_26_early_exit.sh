#!/bin/bash


python3 train_base_channel_skip_new_gate.py train densenet121 --data /nvdatasets/imagenet --resume /tandatasets/jianghao-results/large_minimum/densenet121_skip_channel_new_gate_minimum_66.0/checkpoint_latest.pth.tar --minimum 65 --beta 4e-6 --lr 0.01 --save-folder /tandatasets/jianghao-results/early_exit --batch-size 512

