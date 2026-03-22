from typing import Dict
from app.api.schemas import TaskConfig
from .task.classification import data, train_model
from .task.classification.models import get_model

from app.logs import get_logger

logger = get_logger(__name__)

def training_model(
        config: TaskConfig
    ):
    task_id = config.task_id
    try:
        logger.info(f"💾 Задача[{task_id}]: Старт")

        train_loader, val_loader, test_loader, classes = data.load_dataloaders(
            **config.data_loader_params.model_dump()
        )


        model = get_model(
            **config.model_params.model_dump(),
            num_class = len(classes),
        )

        if config.model_params.weights == False:
            img_w, img_h = model.get_input_size_for_weights()
            config.data_loader_params.img_w_size = img_w
            config.data_loader_params.img_h_size = img_h

        trainer = train_model.Trainer(
            model,
            train_loader,
            val_loader,
            test_loader,
            classes,
            **config.trainer_params.model_dump()
        )

        trainer.train()
    except Exception as e:
        logger.error(e)