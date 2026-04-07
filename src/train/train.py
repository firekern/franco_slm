import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import wandb



def train(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info("Starting training...")

if __name__ == "__main__":
    train()