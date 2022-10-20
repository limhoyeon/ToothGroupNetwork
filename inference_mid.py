import sys
import os
sys.path.append(os.getcwd())
from glob import glob
from inference_pipeline_mid import InferencePipeLine
import argparse
from predict_utils import ScanSegmentation

parser = argparse.ArgumentParser(description='Argparse Tutorial')

parser.add_argument('--input_path', type=str, required=True)
parser.add_argument('--save_path', type=str, required=True)

args = parser.parse_args()

inference_config = {
    "fps_model_info":{
        "model_parameter" :{
            "input_feat": 6,
            "stride": [1, 4, 4, 4, 4],
            "nstride": [2, 2, 2, 2],
            "nsample": [36, 24, 24, 24, 24],
            "blocks": [2, 3, 4, 6, 3],
            "block_num": 5,
            "planes": [32, 64, 128, 256, 512],
            "crop_sample_size": 3072,
        },
        "load_ckpt_path": "ckpts/0707_cosannealing_val"
    },

    "boundary_model_info":{
        "model_parameter":{
            "input_feat": 6,
            "stride": [1, 4, 4, 4, 4],
            "nstride": [2, 2, 2, 2],
            "nsample": [36, 24, 24, 24, 24],
            "blocks": [2, 3, 4, 6, 3],
            "block_num": 5,
            "planes": [32, 64, 128, 256, 512],
            "crop_sample_size": 3072,
        },
        "load_ckpt_path": "ckpts/0711_bd_cbl_aug_test_val"
    },

    "boundary_sampling_info":{
        "bdl_ratio": 0.7,
        "num_of_bdl_points": 20000,
        "num_of_all_points": 24000,
    },

    "orginal_data_obj_path": "G:/tooth_seg/main/all_datas/chl/3D_scans_per_patient_obj_files",
}

stl_path_ls = []
for dir_path in [
    x[0] for x in os.walk(args.input_path)
    ][1:]:
    stl_path_ls += glob(os.path.join(dir_path,"*.obj"))

pred_obj = ScanSegmentation(InferencePipeLine(inference_config))
os.makedirs(args.save_path, exist_ok=True)
for i in range(len(stl_path_ls)):
    print(f"Processing: ", i,":",stl_path_ls[i])
    base_name = os.path.basename(stl_path_ls[i]).split(".")[0]
    pred_obj.process(stl_path_ls[i], os.path.join(args.save_path, os.path.basename(stl_path_ls[i]).replace(".obj", ".json")))