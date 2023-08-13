import argparse
from tqdm import tqdm
import torch
import numpy as np
import os
import json
import torch.nn as nn
import cv2
import time

import wandb

wandb.init(project='I3D features counting')

os.environ['CUDA_VISIBLE_DEVICES'] = '1'                                                                
device = torch.device('cuda:0')
model_name = "i3d_r50"
model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=True)

model.blocks[6] = nn.Sequential(
    nn.AvgPool3d(kernel_size = (4, 7, 7), stride = (1, 1, 1), padding = (0, 0, 0)),
    nn.AdaptiveAvgPool3d(output_size=1)
)

model = model.to(device)
model = model.eval()

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo
)

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
frames_per_second = 30
alpha = 4
max_buff_len = 1000
 
transform = Compose([
    Lambda(lambda x: x/255.0),
    NormalizeVideo(mean, std),
    ShortSideScale(size=side_size),
    CenterCropVideo(crop_size)
])

class VideoWrapper:
    def __init__(self, input_file):
        self.cap = cv2.VideoCapture(input_file)
                        
    def get(self, from_frame, need):
        i = 0
        success = True
        res = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, from_frame)

        while success and i < need:
            success, frame = self.cap.read()
            i += 1
            if success:
                frame = np.transpose(frame, (2, 0, 1))
                res.append(frame)

        res = np.stack(res, axis = 1)
        return res
                                                                                                            
    def num_frames(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))


def count_one_video(input_file, output_file):
    video = VideoWrapper(input_file)
    shift = 0
    buffed_till = 0
    output = []
    num_frames = video.num_frames()

    for start_frame in tqdm(range(num_frames)):
        from_frame = max(0, start_frame - window_side)
        to_frame = min(num_frames - 1, start_frame + window_side + 1)

        if to_frame >= buffed_till:
            buffed_till = min(buffed_till + max_buff_len, num_frames)
            shift = from_frame
            need = buffed_till - shift
            video_buff = transform(torch.tensor(video.get(shift, need)))

        before_padding_sz, after_padding_sz = 0, 0
        if from_frame == 0:
            before_padding_sz = window_side - start_frame
        if to_frame == (num_frames - 1):
           after_padding_sz = window_side + 2 + start_frame - num_frames
                                                    
        buff_shape = video_buff.shape
        before_padding = torch.zeros([buff_shape[0], before_padding_sz, buff_shape[2], buff_shape[3]])
        after_padding = torch.zeros([buff_shape[0], after_padding_sz, buff_shape[2], buff_shape[3]])

        inputs = video_buff[:, (from_frame - shift):(to_frame - shift), :, :]
        inputs = torch.cat([before_padding, inputs, after_padding], 1)
        inputs = [i.to(device)[None, ...] for i in inputs]
        inputs = torch.cat(inputs).unsqueeze(0)

        features = model(inputs)
        output.append(features.detach().cpu().numpy())

    res = np.concatenate(output, axis = 2)
    np.save(output_file, res.squeeze())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default = '/DATA/ichuviliaeva/videos/50salads_vid/rgb/')
    parser.add_argument('--output_dir', default = '/DATA/ichuviliaeva/videos/i3d_experemental/features_diff_pool/')
    args = parser.parse_args()
    for f in os.listdir(args.input_dir):
        if f.split('.')[1] == 'avi':
            print('counting for ' + f)
            start = time.time()
            count_one_video(args.input_dir + f, args.output_dir + f.split('.')[0] + '.npy')
            end = time.time()
            print('ended, time in seconds = ', end - start)

if __name__ == '__main__':
        main()

