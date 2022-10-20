import os
import glob
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--split_target_data_path', required=False, default="example_data/processed_data", type=str, help="data directory to split")
parser.add_argument('--test_split_num', required=True, type=int, help="the number of test data")
parser.add_argument('--save_results_path', required=False, default="example_data/split_info", type=str, help="data path in which split info will be saved")
args = parser.parse_args()

os.makedirs(args.save_results_path, exist_ok=True)

train_list = glob.glob(os.path.join(args.split_target_data_path, "*"))
for split_num in range(4):
    f_train = open(os.path.join(args.save_results_path, "train_fold.txt"), 'w')
    f_test = open(os.path.join(args.save_results_path, "test_fold.txt"), 'w')
    test_idx_ls = np.random.permutation(len(train_list))
    test_idx_ls = test_idx_ls[:args.test_split_num]
    for idx, item in enumerate(train_list):
        if idx in test_idx_ls:
            f_test.write(item)
            f_test.write('\n')
        else:
            f_train.write(item)
            f_train.write('\n')
    f_train.close()
    f_test.close()
