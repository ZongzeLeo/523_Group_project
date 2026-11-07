"""CAMUS ED/ES frames (2CH + 4CH) as a 2D segmentation dataset.

Preprocessing: intensities / 255, then an isotropic, physical-aspect-preserving
letterbox into a size x size grid (images bilinear, masks nearest). Padding is
outside the valid field of view and is ignored by the loss (label IGNORE).
"""
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from common import PHASES, VIEWS, image_path, load_nifti

IGNORE = 255


def letterbox_params(shape, spacing, size):
    """Isotropic target grid: the longer physical side spans `size` pixels."""
    H, W = shape
    ext0, ext1 = H * spacing[0], W * spacing[1]
    iso = max(ext0, ext1) / size                       # mm per letterboxed pixel
    h = min(size, max(1, round(ext0 / iso)))
    w = min(size, max(1, round(ext1 / iso)))
    return {"h": h, "w": w, "top": (size - h) // 2, "left": (size - w) // 2,
            "iso_spacing_mm": iso, "native_shape": (H, W)}


def letterbox(arr, params, size, is_mask=False):
    """Native 2D array -> (size, size) tensor placed in the letterbox."""
    t = torch.from_numpy(arr.astype(np.float32))[None, None]
    if is_mask:
        t = F.interpolate(t, size=(params["h"], params["w"]), mode="nearest-exact")
    else:
        t = F.interpolate(t, size=(params["h"], params["w"]), mode="bilinear",
                          align_corners=False)
    out = torch.zeros(1, 1, size, size)
    out[..., params["top"]:params["top"] + params["h"],
        params["left"]:params["left"] + params["w"]] = t
    return out[0, 0]


def valid_fov(params, size):
    v = torch.zeros(size, size, dtype=torch.uint8)
    v[params["top"]:params["top"] + params["h"], params["left"]:params["left"] + params["w"]] = 1
    return v


def unletterbox_probs(probs, params):
    """(1, C, size, size) probabilities -> (1, C, H, W) at native resolution."""
    crop = probs[..., params["top"]:params["top"] + params["h"],
                 params["left"]:params["left"] + params["w"]]
    return F.interpolate(crop, size=params["native_shape"], mode="bilinear",
                         align_corners=False)


def preprocess_image(arr, spacing, size):
    """Native image -> ((1, size, size) network input, letterbox params)."""
    params = letterbox_params(arr.shape, spacing, size)
    img = letterbox(arr / 255.0, params, size)[None]
    return normalize(img, valid_fov(params, size)), params


def normalize(img, valid):
    """Per-image z-score using valid-FOV pixels only; padding set to 0."""
    v = valid.bool()
    vals = img[0][v]
    out = (img - vals.mean()) / (vals.std() + 1e-6)
    return out * v


def random_affine(img, mask, valid, max_rot=10.0, scale=(0.9, 1.1), max_shift=0.05):
    """Same random rotation/scale/translation for image (bilinear), mask and valid
    FOV (nearest). No flips: flipping would mirror the apical view's anatomy."""
    ang = math.radians(random.uniform(-max_rot, max_rot))
    s = random.uniform(*scale)
    tx, ty = (random.uniform(-max_shift, max_shift) * 2 for _ in range(2))
    cos, sin = math.cos(ang) / s, math.sin(ang) / s
    theta = torch.tensor([[cos, -sin, tx], [sin, cos, ty]], dtype=torch.float32)[None]
    grid = F.affine_grid(theta, (1, 1, *img.shape[-2:]), align_corners=False)
    img = F.grid_sample(img[None], grid, mode="bilinear", padding_mode="zeros",
                        align_corners=False)[0]
    mv = torch.stack([mask, valid]).float()[None]
    mv = F.grid_sample(mv, grid, mode="nearest", padding_mode="zeros",
                       align_corners=False)[0].long()
    return img, mv[0], mv[1]


def random_intensity(img):
    """Gamma, contrast/brightness and Gaussian noise on a [0, 1] image."""
    if random.random() < 0.5:
        img = img.clamp(0, 1) ** random.uniform(0.7, 1.5)
    if random.random() < 0.5:
        img = img * random.uniform(0.85, 1.15) + random.uniform(-0.1, 0.1)
    if random.random() < 0.3:
        img = img + torch.randn_like(img) * random.uniform(0.0, 0.05)
    return img


class CamusFrames(Dataset):
    """All (patient, view, phase) frames for the given patients, preloaded and letterboxed.
    __getitem__ returns (image, mask with IGNORE outside the valid FOV, index)."""

    def __init__(self, data_root, patients, size=256, augment=False):
        self.size, self.augment = size, augment
        self.items, self.images, self.masks, self.valid = [], [], [], []
        for pid in patients:
            for view in VIEWS:
                for phase in PHASES:
                    img, _, spacing = load_nifti(image_path(data_root, pid, view, phase))
                    gt, _, _ = load_nifti(image_path(data_root, pid, view, phase, gt=True))
                    assert img.shape == gt.shape, f"image/mask mismatch {pid} {view} {phase}"
                    p = letterbox_params(img.shape, spacing, size)
                    self.images.append(letterbox(img / 255.0, p, size).half())
                    self.masks.append(letterbox(gt, p, size, is_mask=True).to(torch.uint8))
                    self.valid.append(valid_fov(p, size))
                    self.items.append((pid, view, phase))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img = self.images[i].float()[None]
        mask, valid = self.masks[i].long(), self.valid[i].long()
        if self.augment:
            img, mask, valid = random_affine(img, mask, valid)
            img = random_intensity(img)
        img = normalize(img, valid)
        mask = mask.masked_fill(valid == 0, IGNORE)
        return img, mask, i
