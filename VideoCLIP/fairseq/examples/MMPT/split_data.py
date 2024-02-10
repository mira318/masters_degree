import pandas as pd
import math
from sklearn.model_selection import train_test_split
import argparse
import os
from tqdm import tqdm

def common_column(cat1, cat2):
    if type(cat1) == float:
        cat1 = '_'
    if type(cat2) == float:
        cat2 = '_'
    return cat1 + ' & ' + cat2
       
parser = argparse.ArgumentParser(description="parser to split HowTo100M")
parser.add_argument('--how2csv', default="G:\IChuviliaeva\Data\howTo100M_meta\HowTo100M_v1.csv")
parser.add_argument('--saveto', default="G:\IChuviliaeva\Data\howTo100M_meta")
parser.add_argument('--vfeatdir', default="G:\IChuviliaeva\Data\howto100m_s3d_features\howto100m_s3d_features")
args = parser.parse_args()

how2_df = pd.read_csv(args.how2csv)
how2_df['common'] = how2_df.apply(lambda t: common_column(t['category_1'], t['category_2']), axis = 1)

filtered = []
for i in tqdm(range(len(how2_df))):
    if os.path.isfile(os.path.join(args.vfeatdir, how2_df.iloc[i]['video_id'] + '.npy')):
        filtered.append(how2_df.iloc[i])
        
filtered = pd.DataFrame(filtered)
train, val = train_test_split(filtered, test_size=4000, stratify=filtered['common'], random_state=37)
print('train size = ', len(train['video_id']))
print('val size = ', len(val['video_id']))

with open(os.path.join(args.saveto, 'how2_s3d_train.lst'), 'w') as f:    
    for v_id in train['video_id']:
        f.write(v_id + '\n')
    f.close()

with open(os.path.join(args.saveto, 'how2_s3d_val.lst'), 'w') as f:
    
    for v_id in val['video_id']:
        f.write(v_id + '\n')
    f.close()

