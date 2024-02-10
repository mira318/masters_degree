import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description="parser to redunt .mp4 in naming")
parser.add_argument('--vdir', default="/home/irene/Desktop/mag_degree/how2mini/features")
args = parser.parse_args()

for f in tqdm(os.listdir(args.vdir)):
    splitted = f.split('.')
    os.rename(os.path.join(args.vdir, f), os.path.join(args.vdir, splitted[0] + '.' + splitted[2]))
