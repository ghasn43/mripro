# src/degrade_realistic.py
"""
Cloud-friendly, OpenCV-free "low-field MRI" degradation simulator.

Why this file exists:
- Streamlit Cloud often runs newer Python versions where OpenCV (cv2) wheels may fail to import.
- This implementation avoids cv2 entirely and uses only NumPy + Pillow.

What it simulates (approximate, physics-flavored):
- Bias field (smooth shading)
- Resolution loss via k-space center crop
- Rician-like magnitude noise
- Optional motion-like artifact via k-space phase ramp
- (Warp/distortion is intentionally omitted here to stay cv2-free & stable on Cloud)

Input/Output:
- clean01: float32 array (H,W) in [0,1]
- returns degraded01: float32 array (H,W) in [0,1]
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


# -----------------------------
# Utilities
# -----------------------------
def _normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    lo = np.percentile(x, 1)
    hi = np.percentile(x, 99)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def _is_useful_slice(img01: np.ndarray, min_nonzero_frac: float = 0.02) -> bool:
    # Used by some pipelines to skip blank/background slices
    return float((img01 > 0.05).mean()) >= float(min_nonzero_frac)


def _pil_from01(img01: np.ndarray) -> Image.Image:
    arr = np.clip(img01 * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def _pil_to01(im: Image.Image) -> np.ndarray:
    return (np.asarray(im).astype(np.float32) / 255.0).clip(0.0, 1.0)


def resize01(img01: np.ndarray, size: int) -> np.ndarray:
    """Resize (H,W) -> (size,size) using Pillow."""
    im = _pil_from01(img01)
    im = im.resize((int(size), int(size)), resample=Image.BILINEAR)
    return _pil_to01(im)


def blur01(img01: np.ndarray, radius: float) -> np.ndarray:
    """Gaussian blur using Pillow."""
    im = _pil_from01(img01)
    im = im.filter(ImageFilter.GaussianBlur(radius=float(radius)))
    return _pil_to01(im)


# -----------------------------
# Degradation components
# -----------------------------
def _bias_field(h: int, w: int, strength: float = 0.35) -> np.ndarray:
    """
    Smooth multiplicative shading field, clipped to a reasonable range.

    strength: higher => stronger spatial intensity variation
    """
    xs = np.linspace(-1, 1, w, dtype=np.float32)
    ys = np.linspace(-1, 1, h, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)

    a = np.random.uniform(-strength, strength)
    b = np.random.uniform(-strength, strength)
    c = np.random.uniform(-strength, strength)

    field = 1.0 + a * X + b * Y + c * (X * Y)

    # Smooth heavily with Pillow blur (works without cv2)
    # Convert to 0..1, blur, then scale back around 1.0
    f01 = _normalize01(field)
    # radius chosen based on image size (bigger images => larger blur)
    base = max(h, w)
    radius = np.random.uniform(0.03, 0.09) * base  # e.g. 256 -> ~8..23
    f01 = blur01(f01, radius=radius)

    # Map blurred 0..1 back to approx 0.6..1.5 around 1.0
    field = 0.6 + f01 * (1.5 - 0.6)
    return np.clip(field, 0.6, 1.5).astype(np.float32)


def ifft2c(kspace: np.ndarray) -> np.ndarray:
    """Centered 2D IFFT for last 2 dims."""
    return np.fft.ifftshift(
        np.fft.ifft2(np.fft.fftshift(kspace, axes=(-2, -1)), axes=(-2, -1)),
        axes=(-2, -1),
    )


def _kspace_crop(img01: np.ndarray, crop_frac: float) -> np.ndarray:
    """
    Approximate low-field resolution loss by keeping only the k-space center.
    Smaller crop_frac => blurrier / lower effective resolution.
    """
    img = img01.astype(np.float32, copy=False)
    k = np.fft.fftshift(np.fft.fft2(img))

    H, W = img.shape
    ch = int(H * float(crop_frac))
    cw = int(W * float(crop_frac))
    ch = max(16, ch)
    cw = max(16, cw)

    y0 = (H - ch) // 2
    x0 = (W - cw) // 2

    k2 = np.zeros_like(k)
    k2[y0 : y0 + ch, x0 : x0 + cw] = k[y0 : y0 + ch, x0 : x0 + cw]

    rec = np.abs(np.fft.ifft2(np.fft.ifftshift(k2))).astype(np.float32)
    return _normalize01(rec)


def _rician_noise(img01: np.ndarray, sigma: float) -> np.ndarray:
    """
    Rician-like noise: add Gaussian noise to real/imag then magnitude.
    """
    img = img01.astype(np.float32, copy=False)
    n1 = np.random.normal(0.0, sigma, img.shape).astype(np.float32)
    n2 = np.random.normal(0.0, sigma, img.shape).astype(np.float32)
    mag = np.sqrt((img + n1) ** 2 + (n2) ** 2)
    return np.clip(mag, 0.0, 1.0).astype(np.float32)


def _motion_phase(img01: np.ndarray, severity: float) -> np.ndarray:
    """
    Motion-ish artifact proxy: apply a phase ramp in k-space, causing ghosting-like effects.
    This is a crude but useful approximation for demos/training augmentation.
    """
    img = img01.astype(np.float32, copy=False)
    k = np.fft.fftshift(np.fft.fft2(img))

    H, W = img.shape
    ramp = np.linspace(-np.pi, np.pi, H, dtype=np.float32)
    ramp = ramp * (0.5 + 2.0 * float(severity)) * np.random.uniform(-1.0, 1.0)
    phase = np.exp(1j * ramp)[:, None]

    k2 = k * phase
    rec = np.abs(np.fft.ifft2(np.fft.ifftshift(k2))).astype(np.float32)
    return _normalize01(rec)


# -----------------------------
# Main API
# -----------------------------
def degrade_low_field_realistic(clean01: np.ndarray, severity: float = 0.6) -> np.ndarray:
    """
    Cloud-friendly realistic degradation.
    - severity in [0,1]
    - clean01 expected in [0,1] float, shape (H,W)

    Returns degraded01 in [0,1].
    """
    s = float(np.clip(severity, 0.0, 1.0))
    x = clean01.astype(np.float32, copy=False)
    x = np.clip(x, 0.0, 1.0)

    # 1) Bias/shading (common in low-field / coil sensitivity effects)
    if np.random.rand() < 0.8:
        field = _bias_field(*x.shape, strength=0.20 + 0.45 * s)
        x = np.clip(x * field, 0.0, 1.0)

    # 2) Resolution loss via k-space crop
    crop_frac = float(np.clip(0.95 - 0.55 * s, 0.30, 0.98))
    x = _kspace_crop(x, crop_frac=crop_frac)

    # 3) Occasional motion-like artifacts
    if np.random.rand() < (0.10 + 0.35 * s):
        x = _motion_phase(x, severity=s)

    # 4) Rician-like noise (dominant low-field characteristic)
    sigma = float(0.01 + 0.08 * s)
    x = _rician_noise(x, sigma=sigma)

    return np.clip(x, 0.0, 1.0).astype(np.float32)
