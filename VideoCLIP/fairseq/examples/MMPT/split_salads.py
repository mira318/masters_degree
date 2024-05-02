import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser(description="parser to split 50salads")
parser.add_argument('--saveto', default="G:\\IChuviliaeva\\Data\\50salads_s3d\\meta")
parser.add_argument('--vfeatdir', default="G:\\IChuviliaeva\\Data\\50salads_s3d\\net_features")
args = parser.parse_args()

ids = []
for f in os.listdir(args.vfeatdir):
    if os.path.isfile(os.path.join(args.vfeatdir, f)):
        ids.append(f.split('.')[0])

with open(os.path.join(args.saveto, 'train.lst'), 'w') as f:    
    f.close()

with open(os.path.join(args.saveto, 'val.lst'), 'w') as f:    
    for id in ids:
        f.write(id + '\n')
    f.close()
