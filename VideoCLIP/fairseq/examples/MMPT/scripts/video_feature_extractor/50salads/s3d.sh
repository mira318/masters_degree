#!/bin/bash


python scripts/video_feature_extractor/extract.py \
    --vdir ~/Desktop/mag_degree/50salads_2 \
    --fdir ~/Desktop/mag_degree/feat_50salads_s3d \
    --type=s3d --num_decoding_thread=4 \
    --batch_size 32 --half_precision 1
