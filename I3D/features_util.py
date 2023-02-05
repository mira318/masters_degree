import argparse
from tqdm import tqdm
import torch
import json
import numpy as np
import os
import torch.nn as nn

os.environ['CUDA_VISIBLE_DEVICES'] = '1'                                                                
device = torch.device('cuda:0')
model_name = "i3d_r50"
model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=True)
model.blocks[6] = nn.AdaptiveAvgPool3d(output_size=1)
model = model.to(device)
model = model.eval()

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformCropVideo
)

side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 21
window_side = 10
sampling_rate = 1
frames_per_second = 30
alpha = 4

max_buff_len = 1000
margin = 5

transform = ApplyTransformToKey(
    key = "video", 
    transform = Compose([
        Lambda(lambda x: x/255.0),
        NormalizeVideo(mean, std),
        ShortSideScale(size=side_size),
        CenterCropVideo(crop_size)
    ]),
)

def count_one_video(input_file, output_file):
    video = EncodedVideo.from_path(input_file)
    one_frame_duration = 1 * sampling_rate / frames_per_second
    start_sec = 0
    end_sec = min((max_buff_len + margin) * one_frame_duration, 
                  (video._duration.numerator - 1) * one_frame_duration)
    video_buff = transform(video.get_clip(start_sec = start_sec, end_sec = end_sec))
    inputs_buff = video_buff["video"]
    output = []
    buffed_till = min(max_buff_len, video._duration.numerator)
    shift = 0
    for start_frame in tqdm(range(video._duration.numerator)):
        from_frame = max(0, start_frame - window_side)
        to_frame = min(video._duration.numerator - 1, start_frame + window_side + 1)

        if to_frame >= buffed_till:
            buffed_till = min(buffed_till + max_buff_len, video._duration.numerator)
            shift = from_frame - window_side
            start_sec = shift * one_frame_duration
            end_sec = min(end_sec + ((max_buff_len + margin) * one_frame_duration), 
                          (video._duration.numerator + margin) * one_frame_duration)
            video_buff = transform(video.get_clip(start_sec = start_sec, end_sec = end_sec))
            inputs_buff = video_buff["video"]

        before_padding_sz, after_padding_sz = 0, 0
        if from_frame == 0:
            before_padding_sz = window_side - start_frame
        if to_frame == (video._duration.numerator - 1):
           after_padding_sz = window_side + 2 + start_frame - video._duration.numerator
                                                    
        buff_shape = inputs_buff.shape
        before_padding = torch.zeros([buff_shape[0], before_padding_sz, buff_shape[2], buff_shape[3]])
        after_padding = torch.zeros([buff_shape[0], after_padding_sz, buff_shape[2], buff_shape[3]])
        inputs = inputs_buff[:, (from_frame - shift):(to_frame - shift), :, :]
        inputs = torch.cat([before_padding, inputs, after_padding], 1)
        inputs = [i.to(device)[None, ...] for i in inputs]
        inputs = torch.cat(inputs).unsqueeze(0)
        features = model(inputs)
        output.append(features.detach().cpu().numpy())

    res = np.concatenate(output, axis = 2)
    np.save(output_file, res.squeeze())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default = '/DATA/ichuviliaeva/videos/50salads_vid/rgb/rgb-01-1.avi')
    parser.add_argument('--output', default = '/DATA/ichuviliaeva/videos/i3d_experemental/features/features-01-1.npy')
    args = parser.parse_args()
    count_one_video(args.input, args.output)

if __name__ == '__main__':
        main()

