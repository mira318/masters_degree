#!/bin/bash


python scripts\video_feature_extractor\extract.py \
    --vdir G:\IChuviliaeva\Data\50salads_video\rgb \
    --fdir G:\IChuviliaeva\Data\50salads_s3d_mixed_5c_as_was \
    --type=s3d --num_decoding_thread=4 \
    --batch_size 32 --half_precision 1
