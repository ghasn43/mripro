# src/degrade.py
"""
Compatibility layer.

Your app imports:
    from src.degrade import degrade_low_field

Previously this used OpenCV (cv2). On Streamlit Cloud and many venvs, cv2 isn't available.
So we redirect to the OpenCV-free implementation in src/degrade_realistic.py.
"""

from __future__ import annotations

import numpy as np

# Import the OpenCV-free degradation
from src.degrade_realistic import degrade_low_field_realistic


def degrade_low_field(clean01: np.ndarray, severity: float = 0.6) -> np.ndarray:
    """
    Backwards-compatible function name expected by app.py.

    clean01: float32 (H,W) in [0,1]
    severity: 0..1
    returns: float32 (H,W) in [0,1]
    """
    return degrade_low_field_realistic(clean01, severity=severity)
