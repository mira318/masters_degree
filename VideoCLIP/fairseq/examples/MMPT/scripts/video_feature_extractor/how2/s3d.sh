#!/bin/bash


python ~/Desktop/mag_degree/masters_degree/VideoCLIP/fairseq/examples/MMPT/scripts/video_feature_extractor/extract.py \
    --vdir ~/Desktop/mag_degree/50salads \
    --fdir ~/Desktop/mag_degree/feat_50salads_s3d \
    --type=s3d --num_decoding_thread=4 \
    --batch_size 32 --half_precision 1
