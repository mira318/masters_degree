import os
import argparse

parser = argparse.ArgumentParser(description="parser to redunt .mp4 in naming")
parser.add_argument('--vdir', default="/home/irene/Desktop/mag_degree/how2mini/features")
args = parser.parse_args()

for f in os.listdir(args.vdir):
    splitted = f.split('.')
    os.rename(os.path.join(args.vdir, f), os.path.join(args.vdir, splitted[0] + '.' + splitted[2]))
