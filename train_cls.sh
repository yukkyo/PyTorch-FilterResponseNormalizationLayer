#!/usr/bin/env bash


# Comparing pretrain is useful or not
python train_cls.py --model se_resnext50_32x4d --fp16

python train_cls.py --model se_resnext50_32x4d --fp16 --frn

python train_cls.py --use-pretrain --model se_resnext50_32x4d --fp16

python train_cls.py --use-pretrain --model se_resnext50_32x4d --fp16 --frn

# Comparing Batch Size


# Comparing resnet34 and se-resnext50
