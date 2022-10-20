from models.bdl_grouping_netowrk_model import BdlGroupingNetworkModel
from models.modules.grouping_network_module import GroupingNetworkModule
import os
from runner import runner


exp_name = "1003_bdl_test"
config = {
    #WandB Options
    #If you dont want to use wandb, just set config["wandb"]["wandb_on"] value to False.
    "wandb":{
        "entity": "hoyeon94",
        "wandb_on": True,
        "project": "mesh_seg",
        "tags": "0516_full_crop",
        "notes": "coord부분에 mask만 달았을 때 확인해보기 위함",
        "name": exp_name,
    },


    #Generator options
    #The fps sampled points are input to the model
    "generator":{
        #"train_data_path": f"example_data/split_info/train_fold.txt",
        "train_data_path": f"datasets/data_chl_fps",
        #"val_data_path": f"example_data/split_info/test_fold.txt",
        "val_data_path": f"datasets/data_chl_fps/test",
        "aug_obj_str": "aug.Augmentator([aug.Scaling([0.85, 1.15]), aug.Rotation([-30,30], 'fixed'), aug.Translation([-0.2, 0.2])])",
        "train_batch_size": 1,
        "val_batch_size": 1,
    },
    
    "tr_set":{
        "checkpoint_path": f"ckpts/{exp_name}",
        "optimizer":{
            "lr": 1e-1,
            "NAME": 'sgd',
            "momentum": 0.9,
            "weight_decay": 1.0e-4,
        },
        "scheduler":{
            "sched": 'exp', 
            "schedueler_step": 150,
            "step_decay": 0.999
        },
    },
    
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

    "boundary_sampling_info":{
        "orginal_data_obj_path": "G:/tooth_seg/main/all_datas/chl/3D_scans_per_patient_obj_files",
        "orginal_data_json_path": "G:/tooth_seg/main/all_datas/chl/ground-truth_labels_instances",
        "bdl_cache_path": "bdl_cache/bdl_cache_0930_test_val",
        "bdl_ratio": 0.7,
        "num_of_bdl_points": 20000,
        "num_of_all_points": 24000,
    },
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
        "load_ckpt_path": "ckpts/0930_test_val"
    }
}

model = BdlGroupingNetworkModel(config, GroupingNetworkModule)
runner(config, model, "train")
