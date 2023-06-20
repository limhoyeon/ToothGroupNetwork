from trainer import Trainer
from generator import DentalModelGenerator
from torch.utils.data import DataLoader
import os
import torch

def collate_fn(batch):
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

def get_generator_set(config, is_test=False):
    if not is_test:
        point_loader = DataLoader(
            DentalModelGenerator(
                config["input_data_dir_path"], 
                aug_obj_str=config["aug_obj_str"],
                split_with_txt_path=config["train_data_split_txt_path"]
            ), 
            shuffle=True,
            batch_size=config["train_batch_size"],
            collate_fn=collate_fn
        )

        val_point_loader = DataLoader(
            DentalModelGenerator(
                config["input_data_dir_path"], 
                aug_obj_str=None,
                split_with_txt_path=config["val_data_split_txt_path"]

            ), 
            shuffle=False,
            batch_size=config["val_batch_size"],
            collate_fn= collate_fn
        )
        return [point_loader, val_point_loader]

def runner(config, model):
    gen_set = [get_generator_set(config["generator"], False)]
    print("train_set", len(gen_set[0][0]))
    print("validation_set", len(gen_set[0][1]))
    trainner = Trainer(config=config, model = model, gen_set=gen_set)
    trainner.run()