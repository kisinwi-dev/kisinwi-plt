from app.api.schemes import *
from app.core import training_model

task_config_dict = {
    "task_id": "cat_dog",
    "data_loader_params": {
        "dataset_id": "pet_img",
        "version_id": "v_1",
        "img_w_size": 224,
        "img_h_size": 224,
        "batch_size": 32
    },
    "model_params": {
        "type": "Resnet",
        "name": "resnet50",
        "weights": True
    },
    "trainer_params": {
        "loss_fn_config": {
            "type": "CrossEntropyLoss",
            "params": {
                "reduction": "mean",
                "label_smoothing": 0.1
            }
        },
        "optimizer_config": {
            "type": "AdamW",
            "params": {
                "lr": 0.0001,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.05,
                "amsgrad": False
            }
        },
        "scheduler_config": {
            "type": "CosineAnnealingLR",
            "params": {
                "T_max": 30,
                "eta_min": 1e-6,
                "last_epoch": -1
            }
        },
        "device": "cuda",
        "epochs": 30
    }
}

training_model(
    task_id="id_1",
    config=task_config_dict,
)