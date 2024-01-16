import pandas as pd
import math
from sklearn.model_selection import train_test_split
import argparse
import os
import tqdm

def common_column(cat1, cat2):
    if type(cat1) == float:
        cat1 = '_'
    if type(cat2) == float:
        cat2 = '_'
    return cat1 + ' & ' + cat2
       
parser = argparse.ArgumentParser(description="parser to split HowTo100M")
parser.add_argument('--how2csv', default="~/Desktop/mag_degree/HowTo100M_v1.csv")
parser.add_argument('--saveto', default="/home/irene/Desktop/mag_degree/how2mini")
args = parser.parse_args()

how2_df = pd.read_csv(args.how2csv)
how2_df['common'] = how2_df.apply(lambda t: common_column(t['category_1'], t['category_2']), axis = 1)
train, val = train_test_split(how2_df, test_size=4000, stratify=how2_df['common'], random_state=37)

with open(os.path.join(args.saveto, 'how2_s3d_train.lst'), 'w+') as f:
    print(len(train['video_id']))
    for v_id in train['video_id']:
        f.write(v_id + '.mp4\n')
    f.close()

with open(os.path.join(args.saveto, 'how2_s3d_val.lst'), 'w+') as f:
    for v_id in val['video_id']:
        f.write(v_id + '.mp4\n')
    f.close()

