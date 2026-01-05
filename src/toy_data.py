# src/toy_data.py
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def make_toy_clean_images(n=12, size=128, seed=7):
    rng = np.random.default_rng(seed)
    imgs = []

    for _ in range(n):
        im = Image.new("L", (size, size), color=0)
        draw = ImageDraw.Draw(im)

        cx, cy = size // 2, size // 2
        rx = int(size * rng.uniform(0.25, 0.4))
        ry = int(size * rng.uniform(0.25, 0.4))

        draw.ellipse(
            [cx - rx, cy - ry, cx + rx, cy + ry],
            fill=int(rng.uniform(90, 160)),
        )

        im = im.filter(ImageFilter.GaussianBlur(radius=rng.uniform(4, 10)))

        arr = np.asarray(im).astype(np.float32) / 255.0
        arr = np.clip(arr + rng.normal(0, 0.02, arr.shape), 0, 1)

        imgs.append(arr.astype(np.float32))

    return imgs
