import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from tqdm.auto import tqdm

import os

from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

logger = logging.getLogger(__name__)

# https://colab.research.google.com/drive/1OHPQf3iM9RD9g2wZRTj7nf8fs3pgbnF4?usp=sharing#scrollTo=vFkgAjyMR8fa
# nanochat karpathy

@hydra.main(config_path="../config", config_name="config.yaml")
def prepare_data(cfg: DictConfig):

    def _tokenize_function(examples):
        tokenized = tokenizer(examples["text"], truncation=False)

        eos_id = tokenizer.eos_token_id
        tokenized["input_ids"] = [
            seq + [eos_id] for seq in tokenized["input_ids"]
        ]
        return {'input_ids': tokenized["input_ids"], 'len': [len(seq) for seq in tokenized["input_ids"]]}

    dataset = load_dataset(cfg.datasets.name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.name)
    tokenized_dataset = dataset.map(_tokenize_function, batched=True, num_proc=cfg.datasets.num_workers)

    # create file if not exists

    if not os.path.exists(os.path.join(cfg.datasets.output_dir, cfg.datasets.name.split("/")[1])):
        os.makedirs(os.path.join(cfg.datasets.output_dir, cfg.datasets.name.split("/")[1]))
    else:
        # delete if exists
        for file in os.listdir(os.path.join(cfg.datasets.output_dir, cfg.datasets.name.split("/")[1])):
            os.remove(os.path.join(cfg.datasets.output_dir, cfg.datasets.name.split("/")[1], file))

    for split, ds in tokenized_dataset.items():
        arr_len = np.sum(ds['len'], dtype=np.uint64)
        filename = os.path.join(cfg.datasets.output_dir,  cfg.datasets.name.split("/")[1], f"{split}.bin")
        _dtype = np.uint16
        arr = np.memmap(filename, dtype=_dtype, mode='w+', shape=(arr_len,))

        idx = 0
        for batch_idx in tqdm(range(cfg.data_prep.total_batches), desc=f"Processing {split} split"):
            batch = ds.shard(num_shards=cfg.data_prep.total_batches, index=batch_idx, contiguous=True).with_format("numpy")
            arr_batch = np.concatenate(batch['input_ids'])
            arr[idx:idx+len(arr_batch)] = arr_batch
            idx += len(arr_batch)

        arr.flush()
    
    # total tokens 
    total_tokens = sum(np.sum(ds['len'], dtype=np.uint64) for ds in tokenized_dataset.values())
    logger.info(f"Total tokens in dataset: {total_tokens}")
    logger

if __name__ == "__main__":
    prepare_data()
