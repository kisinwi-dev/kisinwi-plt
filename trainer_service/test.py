from app.api.schemas import *
from app.core import training_model


task_config_example = TaskConfig(
    task_id="cat_dog",
    data_loader_params=DataLoaderParams(
        dataset_id="pet_img",
        version_id="v_1",
        img_w_size=224,
        img_h_size=224,
        batch_size=32
    ),
    model_params=ModelParams(
        type="Resnet",
        name="resnet50",
        weights=True 
    ),
    trainer_params=TrainerParams(
        loss_fn_config=LossFnConfig(
            type="CrossEntropyLoss",
            params={
                "reduction": "mean",
                "label_smoothing": 0.1
            }
        ),
        optimizer_config=OptimizerConfig(
            type="AdamW",
            params={
                "lr": 0.0001,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.05,
                "amsgrad": False
            }
        ),
        scheduler_config=SchedulerConfig(
            type="CosineAnnealingLR",
            params={
                "T_max": 30,
                "eta_min": 1e-6,
                "last_epoch": -1
            }
        ),
        device="cuda",
        epochs=30
    )
)

training_model(
    config= task_config_example,
)