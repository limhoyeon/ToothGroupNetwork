import sys
import os
sys.path.append(os.getcwd())
from glob import glob
from inference_pipeline_final import InferencePipelineFinal
import argparse
from predict_utils import ScanSegmentation
parser = argparse.ArgumentParser(description='Argparse Tutorial')

parser.add_argument('--input_path', type=str, required=True)
parser.add_argument('--save_path', type=str, required=True)

args = parser.parse_args()

stl_path_ls = []
for dir_path in [
    x[0] for x in os.walk(args.input_path)
    ][1:]:
    stl_path_ls += glob(os.path.join(dir_path,"*.obj"))

pred_obj = ScanSegmentation(InferencePipelineFinal())
os.makedirs(args.save_path, exist_ok=True)
for i in range(len(stl_path_ls)):
    print(f"Processing: ", i,":",stl_path_ls[i])
    base_name = os.path.basename(stl_path_ls[i]).split(".")[0]
    pred_obj.process(stl_path_ls[i], os.path.join(args.save_path, os.path.basename(stl_path_ls[i]).replace(".obj", ".json")))