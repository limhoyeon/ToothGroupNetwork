config={
    "tr_set":{
        "optimizer":{
            "lr": 1e-3,
            "NAME": 'adam',
            "weight_decay": 1.0e-4,
        },
        "scheduler":{
            "sched": 'cosine', 
            "warmup_epochs": 0,
            "full_steps": 40,
            "schedueler_step": 15000000,
            "min_lr": 1e-4,
        },
    },
    "pretrained_centroid_model_path": "ckpts/tsegnet_centroid", # pretrained centroid prediction model ckpt path! 
    # If run_tooth_segmentation_module is False, only the centroid prediction module will be trained.
    "run_tooth_segmentation_module": True
}