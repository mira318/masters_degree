import argparse
from tqdm import tqdm
import torch
import json
import numpy as np
import os
import torch.nn as nn

os.environ['CUDA_VISIBLE_DEVICES'] = '3'                                                                
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

transform = Compose([
    Lambda(lambda x: x/255.0),
    NormalizeVideo(mean, std),
    ShortSideScale(size=side_size),
    CenterCropVideo(crop_size)
])

def count_one_video(input_file, output_file):
    video = EncodedVideo.from_path(input_file)
    one_frame_duration = 1 * sampling_rate / frames_per_second
    start_sec = 0
    end_sec = (video._duration.numerator - 1) * one_frame_duration
    video_buff = video.get_clip(start_sec=start_sec, end_sec=end_sec)
    inputs_buff = video_buff["video"]
    output = []
    for start_frame in tqdm(range(video._duration.numerator)):
        from_frame = max(0, start_frame - window_side)
        to_frame = min(video._duration.numerator - 1, start_frame + window_side + 1)
        before_padding_sz, after_padding_sz = 0, 0
        if from_frame == 0:
            before_padding_sz = window_side - start_frame
        if to_frame == (video._duration.numerator - 1):
           after_padding_sz = window_side + 2 + start_frame - video._duration.numerator
                                                    
        buff_shape = inputs_buff.shape
        before_padding = torch.zeros([buff_shape[0], before_padding_sz, buff_shape[2], buff_shape[3]])
        after_padding = torch.zeros([buff_shape[0], after_padding_sz, buff_shape[2], buff_shape[3]])
        inputs = inputs_buff[:, from_frame:to_frame, :, :]
        inputs = torch.cat([before_padding, inputs, after_padding], 1)
        inputs = transform(inputs)
        inputs = [i.to(device)[None, ...] for i in inputs]
        inputs = torch.cat(inputs).unsqueeze(0)
        features = model(inputs)
        output.append(features.detach().cpu().numpy())

    res = np.concatenate(output, axis = 2)
    print(res.squeeze().shape)
    np.save(output_file, res.squeeze())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos_dir', default = '/DATA/ichuviliaeva/videos/50salads_vid/rgb')
    parser.add_argument('--features_dir', default = '/DATA/ichuviliaeva/videos/i3d_experemental/features')
    args = parser.parse_args()
    for f in os.listdir(args.videos_dir):
        print('file ', f)
        if f.split('.')[-1] == 'avi':
            count_one_video(
                    args.videos_dir + '/' + f, 
                    args.features_dir + '/features-' + \
                    f.split('-')[1] + '-' + f.split('-')[2].split('.')[0] + '.npy'
            )

if __name__ == '__main__':
        main()

