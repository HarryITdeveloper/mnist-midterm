from typing import Optional
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw

def _clip_uint8(a: np.ndarray) -> np.ndarray:
    return np.clip(a, 0, 255).astype(np.uint8)

def _rand_affine_on_canvas(img_np: np.ndarray, severity: int, rng: random.Random) -> Image.Image:
    # Place digit on bigger canvas, apply rotation/scale/translation, crop back to 28x28
    pil = Image.fromarray(img_np, mode="L")
    size = 28
    s = 1.0 + rng.uniform(-0.05*severity, 0.05*severity)
    new_size = max(16, min(40, int(round(size * s))))
    pil = pil.resize((new_size, new_size), resample=Image.BILINEAR)

    canvas = Image.new("L", (36, 36), 0)
    cx = (36 - new_size)//2
    cy = (36 - new_size)//2
    canvas.paste(pil, (cx, cy))

    deg = rng.uniform(-6*severity, 6*severity)
    try:
        canvas = canvas.rotate(deg, resample=Image.BILINEAR, expand=False, fillcolor=0)
    except TypeError:
        canvas = canvas.rotate(deg, resample=Image.BILINEAR, expand=False)

    max_shift = min(8, int(2*severity))
    ox = 4 + rng.randint(-max_shift, max_shift)
    oy = 4 + rng.randint(-max_shift, max_shift)
    ox = max(0, min(8, ox))
    oy = max(0, min(8, oy))
    out = canvas.crop((ox, oy, ox+28, oy+28))
    return out

def _jitter_bc(pil: Image.Image, severity: int, rng: random.Random) -> Image.Image:
    if severity <= 0:
        return pil
    c = 1.0 + rng.uniform(-0.15, 0.15) * severity/3.0
    b = 1.0 + rng.uniform(-0.15, 0.15) * severity/3.0
    pil = ImageEnhance.Contrast(pil).enhance(c)
    pil = ImageEnhance.Brightness(pil).enhance(b)
    return pil

def _gaussian_noise(pil: Image.Image, severity: int, rng: random.Random) -> Image.Image:
    if severity <= 0:
        return pil
    arr = np.array(pil).astype(np.float32)
    std = 3.0 * severity
    rs = np.random.RandomState(rng.randint(0, 2**31-1))
    noise = rs.normal(0, std, size=arr.shape)
    arr = arr + noise
    return Image.fromarray(_clip_uint8(arr), mode="L")

def _blur(pil: Image.Image, severity: int, rng: random.Random) -> Image.Image:
    if severity < 3:
        return pil
    radius = 0.2 * severity
    return pil.filter(ImageFilter.GaussianBlur(radius=radius))

def _cutout(pil: Image.Image, severity: int, rng: random.Random) -> Image.Image:
    if severity <= 0:
        return pil
    draw = ImageDraw.Draw(pil)
    holes = max(0, severity // 2)
    for _ in range(holes):
        w = rng.randint(2, 2 + 3*severity)
        h = rng.randint(2, 2 + 3*severity)
        x = rng.randint(0, max(0, 28 - w))
        y = rng.randint(0, max(0, 28 - h))
        draw.rectangle([x, y, x+w, y+h], fill=0)
    return pil

def _occluding_line(pil: Image.Image, severity: int, rng: random.Random) -> Image.Image:
    if severity < 3:
        return pil
    if rng.random() > 0.35 + 0.05*severity:
        return pil
    draw = ImageDraw.Draw(pil)
    x1, y1 = rng.randint(0, 27), rng.randint(0, 27)
    x2, y2 = rng.randint(0, 27), rng.randint(0, 27)
    width = rng.randint(1, max(1, severity//2))
    draw.line([x1, y1, x2, y2], fill=0, width=width)
    return pil

def corrupt_digit(img_np: np.ndarray, severity: int = 0, rng: Optional[random.Random] = None) -> np.ndarray:
    if rng is None:
        rng = random.Random()
    pil = _rand_affine_on_canvas(img_np, severity, rng)
    pil = _jitter_bc(pil, severity, rng)
    pil = _gaussian_noise(pil, severity, rng)
    pil = _blur(pil, severity, rng)
    pil = _occluding_line(pil, severity, rng)
    pil = _cutout(pil, severity, rng)
    return np.array(pil, dtype=np.uint8)
