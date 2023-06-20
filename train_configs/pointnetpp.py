config = {
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
            "min_lr": 1e-5,
        },
    },
}