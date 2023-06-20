import sys
import os
sys.path.append(os.getcwd())
from inference_pipelines.inference_pipeline_maker import make_inference_pipeline
from glob import glob
import argparse
from predict_utils import ScanSegmentation

parser = argparse.ArgumentParser(description='Inference models')
parser.add_argument('--input_dir_path', default="G:/tooth_seg/main/all_datas/chl/3D_scans_per_patient_obj_files", type=str, help = "input directory path that contain obj files.")
parser.add_argument('--split_txt_path', default="base_name_test_fold.txt" ,type=str,help = "split txt path.")
parser.add_argument('--save_path', type=str, default="test_results", help = "result save directory.")
parser.add_argument('--model_name', type=str, default="tgnet", help = "model name. list: tsegnet | tgnet | pointnet | pointnetpp | dgcnn | pointtransformer")
parser.add_argument('--checkpoint_path', default="ckpts/tgnet_fps" ,type=str,help = "checkpoint path.")
parser.add_argument('--checkpoint_path_bdl', default="ckpts/tgnet_bdl" ,type=str,help = "checkpoint path(for tgnet_bdl).")
args = parser.parse_args()

split_base_name_ls = []
if args.split_txt_path != "":
    f = open(args.split_txt_path, 'r')
    while True:
        line = f.readline()
        if not line: break
        split_base_name_ls.append(line.strip())
    f.close()

stl_path_ls = []
for dir_path in [
    x[0] for x in os.walk(args.input_dir_path)
    ][1:]:
    if os.path.basename(dir_path) in split_base_name_ls: 
        stl_path_ls += glob(os.path.join(dir_path,"*.obj"))

pred_obj = ScanSegmentation(make_inference_pipeline(args.model_name, [args.checkpoint_path+".h5", args.checkpoint_path_bdl+".h5"]))
os.makedirs(args.save_path, exist_ok=True)
for i in range(len(stl_path_ls)):
    print(f"Processing: ", i,":",stl_path_ls[i])
    base_name = os.path.basename(stl_path_ls[i]).split(".")[0]
    pred_obj.process(stl_path_ls[i], os.path.join(args.save_path, os.path.basename(stl_path_ls[i]).replace(".obj", ".json")))