"""Create a reproducible visual QC sheet of LV overlays, one patient per split."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion

from .dataset import CamusDataset


def make_preview(prepared_root, output):
    tile_w, tile_h = 280, 300
    sheet = Image.new("RGB", (4 * tile_w, 3 * tile_h), (22, 24, 28))
    draw = ImageDraw.Draw(sheet)
    for row_number, split in enumerate(("train", "validation", "test")):
        dataset = CamusDataset(prepared_root, split)
        patient = next(iter(dataset.patient_metadata))
        samples = dataset.patient_frames(patient, native=False)
        for column, key in enumerate(("2CH_ED", "2CH_ES", "4CH_ED", "4CH_ES")):
            sample = samples[key]
            rgb = np.repeat((sample["image"][0] * 255).astype(np.uint8)[..., None], 3, axis=2)
            mask = sample["mask"].astype(bool)
            boundary = mask & ~binary_erosion(mask)
            rgb[boundary] = [50, 255, 130]
            tile = Image.fromarray(rgb)
            tile.thumbnail((256, 256))
            x, y = column * tile_w, row_number * tile_h
            sheet.paste(tile, (x + (tile_w - tile.width) // 2, y + 34))
            draw.text((x + 12, y + 6), f"{split}: {patient} {key}", fill=(240, 242, 245))
            draw.text((x + 12, y + 20), "LV cavity outline | physical aspect retained", fill=(170, 180, 190))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output", type=Path, default=Path("data/reports/qc_preview.png"))
    args = parser.parse_args()
    print(make_preview(args.prepared_root, args.output).resolve())
