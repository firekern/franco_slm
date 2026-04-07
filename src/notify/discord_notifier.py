import base64
import logging
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
BOT_NAME    = os.getenv("DISCORD_BOT_NAME", "FRANCO")
BOT_AVATAR  = os.getenv("DISCORD_BOT_AVATAR")


def _patch_webhook_avatar(avatar_path: str) -> None:
    """Aggiorna l'avatar permanente del webhook con un file locale."""
    path = Path(avatar_path)
    if not path.exists():
        logger.warning(f"Avatar non trovato: {avatar_path}")
        return

    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode()
    data_uri = f"data:{mime};base64,{data}"

    # estrai id e token dall'URL: .../webhooks/{id}/{token}
    parts = WEBHOOK_URL.rstrip("/").split("/")
    webhook_id, webhook_token = parts[-2], parts[-1]
    api_url = f"https://discord.com/api/v10/webhooks/{webhook_id}/{webhook_token}"

    try:
        resp = httpx.patch(api_url, json={"name": BOT_NAME, "avatar": data_uri}, timeout=10)
        if resp.status_code == 200:
            logger.info("Avatar del webhook aggiornato.")
        else:
            logger.warning(f"Patch avatar fallita: {resp.status_code} {resp.text}")
    except Exception as e:
        logger.warning(f"Patch avatar fallita: {e}")


def setup() -> None:
    """Chiama questa funzione una volta all'avvio del training."""
    if not WEBHOOK_URL or "YOUR_WEBHOOK" in WEBHOOK_URL:
        return
    if BOT_AVATAR and not BOT_AVATAR.startswith("http"):
        _patch_webhook_avatar(BOT_AVATAR)


def _send(payload: dict) -> None:
    if not WEBHOOK_URL or "YOUR_WEBHOOK" in WEBHOOK_URL:
        logger.warning("Discord webhook non configurato, skip notifica.")
        return
    payload["username"] = BOT_NAME
    try:
        httpx.post(WEBHOOK_URL, json=payload, timeout=5)
    except Exception as e:
        logger.warning(f"Discord notify fallita: {e}")


def notify_eval(step: int, max_iters: int, train_loss: float, val_loss: float, lr: float) -> None:
    color = 0x57F287 if val_loss < 3.0 else 0xFEE75C  # verde / giallo
    payload = {
        "embeds": [
            {
                "title": f"📊 FRANCO — eval step {step}/{max_iters}",
                "color": color,
                "fields": [
                    {"name": "Train Loss", "value": f"`{train_loss:.4f}`", "inline": True},
                    {"name": "Val Loss",   "value": f"`{val_loss:.4f}`",   "inline": True},
                    {"name": "LR",         "value": f"`{lr:.2e}`",         "inline": True},
                ],
            }
        ]
    }
    _send(payload)


def notify_startup(cfg, total_params: int, trainable_params: int) -> None:
    m = cfg.model
    t = cfg.train

    arch = (
        "```\n"
        "╔══════════════════════════════════╗\n"
        "║           FRANCO  SLM            ║\n"
        "╠══════════════════════════════════╣\n"
        f"║  Vocab size   {m.vocab_size:>8,}          ║\n"
        f"║  d_model      {m.d_model:>8}          ║\n"
        f"║  n_layers     {m.n_layers:>8}          ║\n"
        f"║  n_heads      {m.n_head:>8}          ║\n"
        f"║  d_ff         {m.d_ff:>8}          ║\n"
        f"║  seq_len      {m.seq_len:>8}          ║\n"
        f"║  dropout      {m.dropout:>8}          ║\n"
        "╠══════════════════════════════════╣\n"
        f"║  Total params   {total_params/1e6:>6.2f} M          ║\n"
        f"║  Trainable      {trainable_params/1e6:>6.2f} M          ║\n"
        "╠══════════════════════════════════╣\n"
        f"║  Batch size   {t.batch_size:>8}          ║\n"
        f"║  Max iters    {t.max_iters:>8,}          ║\n"
        f"║  LR           {t.learning_rate:>8.0e}          ║\n"
        f"║  Warmup       {t.warmup_steps:>8}          ║\n"
        "╚══════════════════════════════════╝\n"
        "```"
    )

    payload = {
        "embeds": [
            {
                "title": "🚀 FRANCO — Training avviato",
                "color": 0xEB459E,
                "description": arch,
                "footer": {"text": "may god have mercy on us"},
            }
        ]
    }
    _send(payload)


def notify_sample(step: int, sample: str) -> None:
    truncated = sample[:1800] + "..." if len(sample) > 1800 else sample
    payload = {
        "embeds": [
            {
                "title": f"✍️ FRANCO genera — step {step}",
                "color": 0x5865F2,
                "description": f"```\n{truncated}\n```",
            }
        ]
    }
    _send(payload)
