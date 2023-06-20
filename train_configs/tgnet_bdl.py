config = {
    "tr_set":{
        "optimizer":{
            "lr": 1e-1,
            "NAME": 'sgd',
            "momentum": 0.9,
            "weight_decay": 1.0e-4,
        },
        "scheduler":{
            "sched": 'cosine', 
            "warmup_epochs": 0,
            "full_steps": 40,
            "schedueler_step": 15000000,
            "min_lr": 1e-5,
        },
        "loss":{
            "cbl_loss_1": 1,
            "cbl_loss_2": 1,
            "tooth_class_loss_1":1,
            "tooth_class_loss_2":1,
            "offset_1_loss": 0.03,
            "offset_1_dir_loss": 0.03,
            "chamf_1_loss": 0.15,
        }
    },
    #Changing the model parameters does not actually alter the model parameters (not implemented).
    "model_parameter":{
        "input_feat": 6,
        "stride": [1, 1],
        "nsample": [36, 24],
        "blocks": [2, 3],
        "block_num": 2,
        "planes": [16, 32],
        "crop_sample_size": 3072,
    },

    "boundary_sampling_info":{
        "orginal_data_obj_path": "G:/tooth_seg/main/all_datas/chl/3D_scans_per_patient_obj_files", # modify this line, original obj data parent path(it`s not preprocessed data path!).
        "orginal_data_json_path": "G:/tooth_seg/main/all_datas/chl/ground-truth_labels_instances", # modify this line, original json data parent path.
        "bdl_cache_path": "temporary_folder", # modify this line, it is just caching folder.
        "bdl_ratio": 0.7,
        "num_of_bdl_points": 20000,
        "num_of_all_points": 24000,
    },
    #Changing the model parameters does not actually alter the model parameters (not implemented).
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
        "load_ckpt_path": "temp_exp_data/ckpts/0215_cbl1_1_cbl2_1_val" # modify this line. the trained checkpoint path of tgnet_fps, eg: ckpts/tgnet_fps.h5!
    }

}