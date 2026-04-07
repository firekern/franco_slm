import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import wandb

from franco.franco import FRANCO
import torch
from pathlib import Path
import numpy as np
from contextlib import nullcontext
from torch.optim.lr_scheduler import LinearLR,SequentialLR, CosineAnnealingLR
from tqdm.auto import tqdm
import torch.nn.functional as F
from transformers import AutoTokenizer
from notify.discord_notifier import notify_eval, notify_sample, notify_startup, setup as discord_setup

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'


# get random batch of data
def get_batch(split: str, cfg: DictConfig):
    out_dir = Path(cfg.datasets.output_dir) / cfg.datasets.name.split("/")[1]

    if split == "train":
        data = np.memmap(out_dir / "train.bin", dtype=np.uint16, mode='r')
    elif split == "val":
        data = np.memmap(out_dir / "validation.bin", dtype=np.uint16, mode='r')

    ix = torch.randint(len(data) - cfg.model.seq_len, (cfg.train.batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+cfg.model.seq_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+cfg.model.seq_len].astype(np.int64)) for i in ix])

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y


@torch.no_grad()
def generate_sample(model, tokenizer, cfg) -> str:
    model.eval()
    tokens = tokenizer.encode(cfg.train.generate_prompt, return_tensors="pt").to(device)

    for _ in range(cfg.train.generate_max_new_tokens):
        tokens_cond = tokens[:, -cfg.model.seq_len:]
        logits, _ = model(tokens_cond)
        logits = logits[:, -1, :] / cfg.train.generate_temperature
        v, _ = torch.topk(logits, min(cfg.train.generate_top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float('-inf')
        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_tok], dim=1)

    return tokenizer.decode(tokens[0].tolist(), skip_special_tokens=True)


def train(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info("Starting training...")

    model = FRANCO(
        vocab_size=cfg.model.vocab_size,
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_head=cfg.model.n_head,
        d_ff=cfg.model.d_ff,
        eps_rms_norm=cfg.model.eps_rms_norm,
        dropout=cfg.model.dropout,
        seq_len=cfg.model.seq_len
    )

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[FRANCO] -> LOCK & LOADED | params: {total_params/1e6:.2f}M ({trainable_params/1e6:.2f}M trainable)")


    learning_rate = cfg.train.learning_rate
    max_iters = cfg.train.max_iters
    warmup_steps = cfg.train.warmup_steps
    min_lr = cfg.train.min_lr
    eval_iters = cfg.train.eval_iters
    batch_size = cfg.train.batch_size
    block_size = cfg.model.seq_len

    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = torch.bfloat16 if dtype == 'bfloat16' else torch.float16
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type, dtype=ptdtype)

    torch.set_default_device(device)
    torch.manual_seed(cfg.train.seed)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=cfg.train.betas, weight_decay=cfg.train.weight_decay, eps=cfg.train.adam_eps)

    scheduler_warmup = LinearLR(optimizer, start_factor=cfg.train.lr_decay_start_factor, total_iters=warmup_steps)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=max_iters - warmup_steps, eta_min=min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_steps])

    # https://stackoverflow.com/questions/72534859/is-gradscaler-necessary-with-mixed-precision-training-with-pytorch
    scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))

    best_val_loss = float('inf')
    best_model_params_path = cfg.train.best_model_params_path

    discord_setup()
    notify_startup(cfg, total_params, trainable_params)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.name)

    for step in tqdm(range(max_iters), desc="Training"):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        x, y = get_batch("train", cfg)
        with ctx:
            logits, loss = model(x, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % eval_iters == 0:
            model.eval()
            with torch.no_grad():
                x_val, y_val = get_batch("val", cfg)
                _, val_loss = model(x_val, y_val)

            train_loss = loss.item()
            val_loss = val_loss.item()
            current_lr = scheduler.get_last_lr()[0]

            logger.info(f"step {step}/{max_iters} | train={train_loss:.4f} | val={val_loss:.4f} | lr={current_lr:.2e}")
            wandb.log({"train/loss": train_loss, "val/loss": val_loss, "lr": current_lr}, step=step)
            notify_eval(step, max_iters, train_loss, val_loss, current_lr)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                Path(best_model_params_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), best_model_params_path)
                logger.info(f"  -> best model salvato (val={best_val_loss:.4f})")

        if step > 0 and step % cfg.train.generate_every == 0:
            sample = generate_sample(model, tokenizer, cfg)
            logger.info(f"\n{'='*50}\nSAMPLE @ step {step}:\n{sample}\n{'='*50}")
            wandb.log({"sample": wandb.Html(f"<pre>{sample}</pre>")}, step=step)
            notify_sample(step, sample)
            model.train()


if __name__ == "__main__":
    train()