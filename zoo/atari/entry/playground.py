#import wandb
import torch

from zoo.atari.config.atari_muzero_rnn_fullobs_config import device

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Device 'cuda'")
    else:
        device = 'cpu'
        print(f"Device: 'cpu'")

    # config = {'device': device}
    # with wandb.init(project="iris-muzero", name="iris-muzero-playground", config=config):
    #     metrics = {"loss": 0.5, "accuracy": 0.6}
    #     wandb.log(data=metrics)