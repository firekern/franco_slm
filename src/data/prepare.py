import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import tqdm
import os

from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

logger = logging.getLogger(__name__)

# https://huggingface.co/datasets/markod0925/TinyStories-Italian/viewer/default/train?row=0

@hydra.main(config_path="../config", config_name="config.yaml")
def prepare(cfg: DictConfig):
    
    dataset = load_dataset(cfg.dataset.name)

    # cut the dataset to the first 100k samples for faster processing
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.name)

    def _tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=cfg.tokenizer.max_length)

    tokenized_dataset = dataset.map(_tokenize_function, batched=True, num_proc=cfg.num_cores, desc="Tokenizing dataset")
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])

    for split, ds in tokenized_dataset.items():
        path = f"{cfg.output_dir}/{cfg.dataset.name}/{split}.bin"
        os.makedirs(os.path.dirname(path), exist_ok=True)

        tot_tokens = sum(len(x["input_ids"]) for x in ds)
        logger.info(f"Saving {split} split with {len(ds)} samples and {tot_tokens} tokens to {path}")

        arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(tot_tokens, ))

        idx = 0
        for seq in tqdm.tqdm(ds["input_ids"], desc=f"Saving {split} split"):
            arr[idx:idx+len(seq)] = seq
            idx += len(seq)

        arr.flush()

    logger.info("We chill babe.")


if __name__ == "__main__":
    prepare()
