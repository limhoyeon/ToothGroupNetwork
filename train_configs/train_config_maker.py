import importlib.util
import sys
import os

def get_default_config(experiment_name, input_data_dir_path, train_data_split_txt_path, val_data_split_txt_path):
    config = {
        #WandB Options
        #If you dont want to use wandb, just set config["wandb"]["wandb_on"] value to False.
        "wandb":{
            "entity": "hoyeon94", #change to your username
            "wandb_on": True,
            "project": "3d tooth mesh segmentation", 
            "tags": "3d tooth mesh segmentation",
            "notes": "3d tooth mesh segmentation",
            "name": experiment_name,
        },
        #Generator options
        #The fps sampled points are input to the model
        "generator":{
            "input_data_dir_path": f"{input_data_dir_path}",
            "train_data_split_txt_path": f"{train_data_split_txt_path}",
            "val_data_split_txt_path": f"{val_data_split_txt_path}",
            "aug_obj_str": "aug.Augmentator([aug.Scaling([0.85, 1.15]), aug.Rotation([-30,30], 'fixed'), aug.Translation([-0.2, 0.2])])",
            "train_batch_size": 1,
            "val_batch_size": 1,
        },
        "checkpoint_path": f"ckpts/{experiment_name}",
    }
    return config

def get_train_config(
        config_path, 
        experiment_name,
        input_data_dir_path,
        train_data_split_txt_path, 
        val_data_split_txt_path,
    ):
    config = {}
    config.update(get_default_config(
        experiment_name, 
        input_data_dir_path, 
        train_data_split_txt_path,
        val_data_split_txt_path
    ))
        
    spec = importlib.util.spec_from_file_location("module.name", config_path)
    loaded_model_config = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = loaded_model_config
    spec.loader.exec_module(loaded_model_config)

    config.update(loaded_model_config.config)
    return config
