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

    from tentamen.model import GRUmodel, Linear
    from tentamen.settings import LinearConfig, GruConfig

    configs = [
        LinearConfig(
            input=13, output=20, tunedir=presets.logdir, h1=100, h2=10, dropout=0.5
        ),
        GruConfig(
            input=13, output=10,tunedir=presets.logdir, num_layers=2, hidden_size=32, dropout=0.2
        )
    ]

    config =  GruConfig(input=13, output=10,tunedir=presets.logdir, num_layers=2, hidden_size=32, dropout=0.2)

    config_GRU = {
        "input_size": 13,
        "hidden_size": 128,
        "dropout": 0.2,
        "num_layers": 3,
        "output_size": 32,
        "num_classes": 20
    }

    model_gru = GRUmodel(config_GRU)
  #  for config in configs:
      #  model = Linear(config.dict())  # type: ignore
       # model_gru = GRUmodel(config.dict())

    trainedmodel = trainloop(
            epochs=20,
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
