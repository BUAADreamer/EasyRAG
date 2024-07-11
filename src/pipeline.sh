#!/bin/bash
python preprocess_zedx.py
CUDA_VISIBLE_DEVICES=0 python main.py