from trainer import Trainer
from generator import DentalModelGenerator
from torch.utils.data import Dataset, DataLoader
import os
from glob import glob
import copy
import torch

def collate_fn(batch):
    num_of_item = len(batch[0])
    output = {}

    for batch_item in batch:
        for key in batch_item.keys():
            if key not in output:
                output[key] = []
            output[key].append(batch_item[key])
    
    for output_key in output.keys():
        if output_key in ["feat", "gt_seg_label", "uniform_feat", "uniform_gt_seg_label"]:
            output[output_key] = torch.stack(output[output_key])
    return output

def get_mesh_path(basename):
    case_name = basename.split("_")[0]
    file_name = basename.split("_")[0]+"_"+basename.split("_")[1]+".obj"
    return os.path.join("all_datas", "chl", "3D_scans_per_patient_obj_files", f"{case_name}", file_name)

def get_generator_set(config):
    point_loader = DataLoader(
        DentalModelGenerator(
            config["train_data_path"], 
            aug_obj_str=config["aug_obj_str"]
        ), 
        shuffle=True,
        batch_size=config["train_batch_size"],
        collate_fn=collate_fn
    )

    val_point_loader = DataLoader(
        DentalModelGenerator(
            config["val_data_path"], 
            aug_obj_str=None
        ), 
        shuffle=False,
        batch_size=config["val_batch_size"],
        collate_fn= collate_fn
    )
    return point_loader, val_point_loader

def runner(config, model, phase):
    gen_set = [get_generator_set(config["generator"])]
    
    if phase=="train":
        trainner = Trainer(config=config, model = model, gen_set=gen_set)
        trainner.run()
    elif phase=="test":
        #DEPR
        pass
