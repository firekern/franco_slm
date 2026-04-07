import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import wandb 

from data.prepare import prepare_data
from train import train 
#from franco import franco

logger = logging.getLogger(__name__)
stages = ["data_prep", "train", "evaluate"]

def init_wandb_for_stage(stage_name: str, cfg: DictConfig):

    # convert cfg to dict for wandb
    dic = OmegaConf.to_container(cfg, resolve=True)

    if stage_name == "data_prep":
        wandb.init(project=cfg.wandb.project, name=f"{cfg.wandb.run_name}_data_prep_{cfg.wandb.comment}", config=dic, group=stage_name, reinit=True)
        prepare_data(cfg)
    elif stage_name == "train":
        wandb.init(project=cfg.wandb.project, name=f"{cfg.wandb.run_name}_train_{cfg.wandb.comment}", config=dic, group=stage_name, reinit=True)
        train.train(cfg)
    elif stage_name == "evaluate":
        wandb.init(project=cfg.wandb.project, name=f"{cfg.wandb.run_name}_evaluate_{cfg.wandb.comment}", config=dic, group=stage_name, reinit=True)
        #franco(cfg)

    wandb.finish()


@hydra.main(config_path="./config", config_name="config.yaml")
def main(cfg: DictConfig):

    logger.info("[STARTING]")
    for stage in stages: 
        if cfg.stages[stage]:
            init_wandb_for_stage(stage, cfg)
    
    wandb.finish()

if __name__ == "__main__":
    main()