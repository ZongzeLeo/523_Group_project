"""Frozen-checkpoint inference; expert target arrays are never loaded here."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform

import numpy as np
import scipy
import torch

from data.download_camus import atomic_json
from data.prepare import write_csv
from data.transforms import restore_to_native
from .io import PreparedFrames, sha256
from .metrics import clean_mask
from .model import UNet
from .training import seed_everything
from data.protocol import identity


def verify_model(run):
    run = Path(run)
    metadata = json.loads((run / "run_metadata.json").read_text())
    if metadata["status"] != "complete" or metadata["smoke_steps"]:
        raise ValueError("Only a completed full training run can be used for evaluation")
    for name, digest in metadata["artifact_sha256"].items():
        if sha256(run / name) != digest:
            raise ValueError(f"Training artifact changed: {name}")
    if sha256(run / "weights/best.pt") != metadata["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed")
    if sha256(Path(__file__).with_name("model.py")) != metadata["code_sha256"]["segmentation/model.py"]:
        compatibility = json.loads((run / "model_compatibility.json").read_text())
        if (compatibility["status"] != "verified"
                or compatibility["checkpoint_sha256"] != metadata["checkpoint_sha256"]
                or compatibility["training_model_sha256"] != metadata["code_sha256"]["segmentation/model.py"]
                or compatibility["packaged_model_sha256"] != sha256(Path(__file__).with_name("model.py"))
                or compatibility["identical_native_prediction_frames"] != 200
                or compatibility["max_probability_difference"] != 0):
            raise ValueError("Model implementation lacks verified packaging equivalence")
    return metadata


def infer(prepared, run, output, split="test", device="cuda"):
    prepared, run, output = Path(prepared), Path(run), Path(output)
    metadata = verify_model(run)
    if sha256(prepared / "run_metadata.json") != metadata["prepared_metadata_sha256"]:
        expected = metadata.get("data_identity")
        if expected is None:
            expected = json.loads((run / "model_compatibility.json").read_text())["data_identity"]
        if identity(prepared) != expected:
            raise ValueError("Inference data do not match training data provenance")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Prediction output is nonempty; choose a new output directory")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA interpreter required, or explicitly choose --device cpu")
    dataset = PreparedFrames(prepared, split, load_targets=False)
    if split == "test" and set(dataset.patient_metadata) & set(metadata["train_patients"]+metadata["validation_patients"]):
        raise ValueError("Test patient overlaps training or checkpoint-selection patients")
    config = json.loads((run / "config.json").read_text())
    seed_everything(config["seed"])
    checkpoint = torch.load(run / "weights/best.pt", map_location=device, weights_only=True)
    model = UNet(config["base_channels"], config.get("normalization", "group")).to(device=device, memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    (output / "arrays").mkdir(parents=True)
    records = []
    with torch.inference_mode():
        for start in range(0, len(dataset), config["batch_size"]):
            indices = list(range(start, min(start+config["batch_size"], len(dataset))))
            samples = [dataset[index] for index in indices]
            images = torch.from_numpy(np.stack([sample["image"] for sample in samples])).to(device=device, memory_format=torch.channels_last)
            with torch.amp.autocast(device_type=device, enabled=device == "cuda"):
                logits = model(images)
            probabilities = logits.float().sigmoid().cpu().numpy()[:, 0]
            for index, sample, probability in zip(indices, samples, probabilities):
                probability[~sample["valid_region"].astype(bool)] = 0
                transform = dataset.transform(index)
                native_probability = restore_to_native(probability, transform)
                raw = (native_probability > config["threshold"]).astype(np.uint8)
                cleaned, quality = clean_mask(raw)
                row = dataset.frames[index]
                relative = f"arrays/{row['frame_id']}.npz"
                np.savez_compressed(output / relative, probability=native_probability.astype(np.float32), raw_mask=raw, mask=cleaned)
                records.append({"frame_id": row["frame_id"], "patient_id": row["patient_id"], "split": split,
                                "view": row["view"], "phase": row["phase"], "path": relative,
                                "sha256": sha256(output / relative), "native_height": native_probability.shape[0],
                                "native_width": native_probability.shape[1], **quality})
    write_csv(output / "predictions.csv", records)
    atomic_json(output / "run_metadata.json", {
        "status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(), "split": split,
        "patients": sorted(dataset.patient_metadata), "frame_count": len(records),
        "training_metadata_sha256": sha256(run / "run_metadata.json"), "checkpoint_sha256": metadata["checkpoint_sha256"],
        "prepared_metadata_sha256": sha256(prepared / "run_metadata.json"),
        "training_run_relative": Path(os.path.relpath(run, output)).as_posix(),
        "selected_epoch": metadata["best_epoch"], "threshold": config["threshold"],
        "postprocessing": config["postprocessing"], "expert_target_arrays_loaded": False,
        "prediction_manifest_sha256": sha256(output / "predictions.csv"),
        "code_sha256": {str(path): sha256(path) for path in [Path("segmentation/infer.py"), Path("segmentation/model.py"),
                          Path("segmentation/metrics.py"), Path("segmentation/io.py"), Path("data/transforms.py")]},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__, "cuda": torch.version.cuda}})
    print(f"Saved {len(records)} native predictions for {len(dataset.patient_metadata)} {split} patients.", flush=True)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared"))
    parser.add_argument("--run", type=Path, default=Path("segmentation/model_artifact"))
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions"))
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    infer(args.prepared, args.run, args.output, args.split, args.device)
