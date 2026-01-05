# src/toy_pretrain.py
"""
Toy checkpoint creator for the demo (OpenCV-free).

Creates a lightweight checkpoint instantly so the Streamlit demo can run
without lengthy training. NOT a clinically meaningful model.
"""

from __future__ import annotations

from pathlib import Path
import torch


def _ckpt_path() -> Path:
    # Keep in project folder so Streamlit Cloud can write it
    return Path("checkpoints") / "toy_unet_tiny.pt"


def make_checkpoint(force: bool = False) -> str:
    """
    Ensure a toy checkpoint exists and return its path as a string.
    """
    ckpt = _ckpt_path()
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    if ckpt.exists() and not force:
        return str(ckpt)

    # Import here to avoid circular imports at module load time
    from src.model import UNetTiny  # noqa: WPS433

    torch.manual_seed(7)
    model = UNetTiny(base=8, p_drop=0.15)
    model.eval()

    payload = {
        "model_state": model.state_dict(),
        "meta": {
            "note": "Toy randomly initialized weights for demo only (not clinical).",
            "seed": 7,
            "arch": "UNetTiny(base=8,p_drop=0.15)",
        },
    }

    torch.save(payload, ckpt)
    return str(ckpt)
