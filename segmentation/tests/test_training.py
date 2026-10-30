import json
from pathlib import Path

import torch
import hashlib
import pytest

from data.prepare import prepare
from data.tests.test_pipeline import source
from segmentation.io import PreparedFrames, sha256

from segmentation.model import UNet
from segmentation.training import augment, segmentation_loss, seed_everything


def test_unet_shape_and_gradient_on_nonmultiple_dimensions():
    torch.set_num_threads(2)
    model = UNet(base_channels=4, normalization="group")
    logits = model(torch.rand(2, 1, 33, 35))
    assert logits.shape == (2, 1, 33, 35)
    target = torch.zeros_like(logits)
    target[:, :, 8:24, 8:24] = 1
    loss = segmentation_loss(logits, target, torch.ones_like(target))
    loss.backward()
    assert torch.isfinite(loss) and model.output.weight.grad.abs().sum() > 0


def test_loss_ignores_padding_and_rewards_correct_mask():
    target = torch.zeros(2, 1, 32, 32)
    target[:, :, 8:24, 8:24] = 1
    valid = torch.zeros_like(target)
    valid[:, :, 4:28, 4:28] = 1
    logits = (target*2-1)*8
    altered = logits.clone()
    altered[~valid.bool()] = 1000
    assert torch.equal(segmentation_loss(logits, target, valid), segmentation_loss(altered, target, valid))
    assert segmentation_loss(logits, target, valid) < .001
    assert segmentation_loss(-logits, target, valid) > 1


def test_geometric_augmentation_keeps_mask_and_valid_region_aligned():
    seed_everything(523)
    target = torch.zeros(2, 1, 32, 32)
    target[:, :, 8:24, 8:24] = 1
    config = json.loads(Path('segmentation/configs/groupnorm.json').read_text())["augmentation"]
    x, y, v = augment(target, target, torch.ones_like(target), config)
    assert x.shape == y.shape == v.shape == target.shape
    assert set(y.unique().tolist()) <= {0, 1}
    assert not (y.bool() & ~v.bool()).any()
    assert ((x > .5) == y.bool()).float().mean() > .98


def test_portable_loader_and_image_only_inference(source, tmp_path):
    root = tmp_path / "prepared"
    prepare(source, root, (32, 32), expected_patients=3)
    labeled = PreparedFrames(root, "train")
    images = PreparedFrames(root, "train", load_targets=False)
    assert "mask" in labeled[0] and "mask" not in images[0]
    assert (labeled[0]["image"] == images[0]["image"]).all()
    path = root / "frames.csv"
    assert sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()
