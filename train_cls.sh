#!/usr/bin/env bash

# Comparing basic experiments
#python train_cls.py --model se_resnext50_32x4d --use-pretrain --fp16
python train_cls.py --frn

# Comparing resnet34 and se-resnext50
#python train_cls.py --model resnet34 --use-pretrain --fp16
#python train_cls.py --model resnet34 --use-pretrain --fp16 --frn
