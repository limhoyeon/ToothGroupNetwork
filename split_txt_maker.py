import os
import glob
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Inference models')
parser.add_argument('--input_dir_path', type=str, default="data_preprocessed_path", help = "input directory path that contain obj files.")
parser.add_argument('--split_txt_save_dir_path', type=str, default="test", help = "split txt path.")
parser.add_argument('--split_ratio_train', type=float, default=0.8, help = "split_ratio for train")
parser.add_argument('--split_ratio_val', type=float, default=0.1, help = "split_ratio for val")
parser.add_argument('--split_ratio_test', type=float, default=0.1, help = "split_ratio for test")
args = parser.parse_args()

all_path_ls = []
for item_path in glob.glob(os.path.join(args.input_dir_path, "*.npy")):
    p_id = os.path.basename(item_path).split("_")[0]
    if p_id not in all_path_ls:
        all_path_ls.append(p_id)
all_path_ls = np.array(all_path_ls)
all_path_ls = all_path_ls[np.random.permutation(all_path_ls.shape[0])]

args.split_ratio = [args.split_ratio_train, args.split_ratio_val, args.split_ratio_test]

if sum(args.split_ratio)!=1:
    raise "error for split ratio"

train_num = int(args.split_ratio[0] * len(all_path_ls))
val_num = int(args.split_ratio[1] * len(all_path_ls))
test_num = int(args.split_ratio[2] * len(all_path_ls))

train_ls = all_path_ls[:train_num]
val_ls = all_path_ls[train_num:train_num+val_num]
test_ls = all_path_ls[train_num+val_num:]

f_train = open(os.path.join(args.split_txt_save_dir_path, f"base_name_train_fold.txt"), 'w')
f_val = open(os.path.join(args.split_txt_save_dir_path, f"base_name_val_fold.txt"), 'w')
f_test = open(os.path.join(args.split_txt_save_dir_path, f"base_name_test_fold.txt"), 'w')

for item in train_ls:
    f_train.write(item + "\n")
for item in val_ls:
    f_val.write(item + "\n")
for item in test_ls:
    f_test.write(item + "\n")
f_train.close()
f_val.close()
f_test.close()