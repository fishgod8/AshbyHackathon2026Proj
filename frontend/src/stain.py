"""Reinhard color normalization (match LAB mean and std to a reference), numpy only.

Slides are scanned, BACH photos come from a microscope camera: the colors differ, and QuiltNet embeddings shift
with color. Matching each slide patch's color statistics to the photo patches is a standard, model-free fix.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

_M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    c = rgb.astype(np.float64) / 255
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ _M.T / _WHITE
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], axis=-1)
    xyz = np.where(f ** 3 > 216 / 24389, f ** 3, (116 * f - 16) / (24389 / 27)) * _WHITE
    c = xyz @ np.linalg.inv(_M).T
    c = np.where(c > 0.0031308, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055, 12.92 * c)
    return (np.clip(c, 0, 1) * 255).round().astype(np.uint8)


def lab_stats(images: list[Image.Image]) -> tuple[np.ndarray, np.ndarray]:
    lab = np.concatenate([rgb_to_lab(np.asarray(im.convert("RGB"))).reshape(-1, 3) for im in images])
    return lab.mean(axis=0), lab.std(axis=0)


def reinhard(im: Image.Image, ref_mean: np.ndarray, ref_std: np.ndarray) -> Image.Image:
    lab = rgb_to_lab(np.asarray(im.convert("RGB")))
    mu, sd = lab.reshape(-1, 3).mean(axis=0), lab.reshape(-1, 3).std(axis=0) + 1e-6
    return Image.fromarray(lab_to_rgb((lab - mu) / sd * ref_std + ref_mean))
