import os
import numpy as np

features_dir = '/DATA/ichuviliaeva/videos/i3d_experemental/features_diff_pool/'
gt_dir = '/DATA/ichuviliaeva/videos/data/50salads/groundTruth/'
new_features_dir = '/DATA/ichuviliaeva/videos/i3d_experemental/features_cut/'
i = 0
for f in os.listdir(features_dir):
    print(f)
    features = np.load(features_dir + f)
    with open(gt_dir + f.replace('npy', 'txt')) as g:
        gt = g.readlines()
        print(features.shape[1], '\t', len(gt))
        if features.shape[1] < len(gt):
            raise Exception('Less features than in ground truth')
        else:
            # diff = features.shape[1] - len(gt)
            # cut_front = diff // 2
            features_cut = features[:, :len(gt)]
            np.save(new_features_dir + f, features_cut)
