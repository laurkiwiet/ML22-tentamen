from datetime import datetime

import torch
from loguru import logger

from tentamen.data import datasets
from tentamen.model import Accuracy
from tentamen.settings import presets
from tentamen.train import trainloop

if __name__ == "__main__":
    logger.add(presets.logdir / "01.log")

    trainstreamer, teststreamer = datasets.get_arabic(presets)
    from tentamen.model import GRUmodel
    from tentamen.settings import GruConfig

    configs = [
        GruConfig(
            input=13,
            output=20,
            tunedir=presets.logdir,
            num_layers=3,
            hidden_size=32,
            dropout=0.2,
        )
    ]
    for config in configs:
        model_gru = GRUmodel(config.dict())  # type: ignore
    trainedmodel = trainloop(
        epochs=50,
        model=model_gru,  # type: ignore
        optimizer=torch.optim.Adam,
        learning_rate=1e-3,
        loss_fn=torch.nn.CrossEntropyLoss(),
        metrics=[Accuracy()],
        train_dataloader=trainstreamer.stream(),
        test_dataloader=teststreamer.stream(),
        log_dir=presets.logdir,
        train_steps=len(trainstreamer),
        eval_steps=len(teststreamer),
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = presets.modeldir / (timestamp + presets.modelname)
    logger.info(f"save model to {path}")
    torch.save(trainedmodel, path)
