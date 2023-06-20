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
            "tooth_class_loss_1":1, #1
        }
    },
    "model_parameter":{
        "input_feat": 6,
        "stride": [1, 4, 4, 4, 4],
        "nsample": [36, 24, 24, 24, 24],
        "blocks": [2, 3, 4, 6, 3],
        "block_num": 5,
        "planes": [32, 64, 128, 256, 512],
        "crop_sample_size": 3072,
    },
}