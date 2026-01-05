# src/infer.py
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def mc_dropout_predict(model, x, n_samples: int = 24, device: str | None = None):
    """
    MC Dropout prediction.

    x can be:
      - numpy.ndarray (H,W) float in [0,1]
      - torch.Tensor shaped (1,1,H,W) or (H,W)
    Returns:
      mean01: numpy (H,W) float32 in [0,1]
      unc:    numpy (H,W) float32 (pixelwise std)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    # Convert numpy -> torch tensor (1,1,H,W)
    if isinstance(x, np.ndarray):
        x = x.astype(np.float32, copy=False)
        if x.ndim != 2:
            raise ValueError(f"Expected numpy (H,W), got {x.shape}")
        x_t = torch.from_numpy(x)[None, None, ...]
    elif torch.is_tensor(x):
        x_t = x
        if x_t.ndim == 2:
            x_t = x_t[None, None, ...]
        elif x_t.ndim == 3:
            x_t = x_t[None, ...]
    else:
        raise TypeError("x must be numpy.ndarray or torch.Tensor")

    x_t = x_t.to(device)

    # Enable dropout for MC sampling
    model.train()

    preds = []
    for _ in range(int(n_samples)):
        y = model(x_t)  # expected (1,1,H,W)
        preds.append(y)

    stack = torch.stack(preds, dim=0)  # (S,1,1,H,W)
    mean = stack.mean(dim=0).squeeze(0).squeeze(0)  # (H,W)
    std = stack.std(dim=0).squeeze(0).squeeze(0)    # (H,W)

    mean01 = mean.detach().cpu().numpy().astype(np.float32)
    unc = std.detach().cpu().numpy().astype(np.float32)

    mean01 = np.clip(mean01, 0.0, 1.0)
    return mean01, unc
