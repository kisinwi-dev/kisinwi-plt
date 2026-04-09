from typing import Dict
from .task.classification import data, train_model
from .task.classification.models import get_model

from app.logs import get_logger

logger = get_logger(__name__)

def training_model(
        task_id: str,
        config: dict
    ):
    try:
        logger.info(f"💾 Задача[{task_id}]: Старт")

        data_loader_params = config["data_loader_params"]

        train_loader, val_loader, test_loader, classes = data.load_dataloaders(
            dataset_id=data_loader_params['dataset_id'],
            version_id=data_loader_params['version_id'],
            img_w_size=data_loader_params['img_w_size'],
            img_h_size=data_loader_params['img_h_size'],
            batch_size=data_loader_params['batch_size']
        )


        model_params = config["model_params"]

        model = get_model(
            type=model_params["type"],
            name=model_params["name"],
            weights=model_params["weights"],
            num_class = len(classes),
        )

        if config["model_params"]["weights"] == False:
            img_w, img_h = model.get_input_size_for_weights()
            config["data_loader_params"]["img_w_size"] = img_w
            config["data_loader_params"]["img_h_size"] = img_h

        trainer_params = config["trainer_params"]
        trainer = train_model.Trainer(
            model,
            train_loader,
            val_loader,
            test_loader,
            classes,
            loss_fn_config=trainer_params["loss_fn_config"],
            optimizer_config=trainer_params["optimizer_config"],
            # scheduler_config=trainer_params["scheduler_config"],
            device=trainer_params["device"],
            epochs=2 # trainer_params["epochs"],
        )

        trainer.train()
    except Exception as e:
        logger.error(e)
        raise