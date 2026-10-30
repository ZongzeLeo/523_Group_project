"""Regenerate publication figures from verified patient and frame measurements."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from data.dataset import CamusDataset
from data.download_camus import atomic_json, sha256
from data.protocol import verify_results
from .run import read_csv
from .statistics import agreement

BLUE, GOLD, CYAN = "#24658A", "#D89020", "#56D2E6"
VIEWS = ("2CH_ED", "2CH_ES", "4CH_ED", "4CH_ES")


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.15)
    ax.set_axisbelow(True)


def identity_scatter(ax, reference, predicted, title, ylabel="Estimated EF (%)"):
    ax.scatter(reference, predicted, color=BLUE, s=25, alpha=.8, edgecolors="white", linewidths=.3)
    ax.plot([0, 100], [0, 100], "--", color=".45", linewidth=1)
    ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="CAMUS reference EF (%)", ylabel=ylabel, title=title)
    ax.set_aspect("equal", adjustable="box")
    style(ax)


def overlay(ax, frame, mask, title):
    ax.imshow(frame["image"], cmap="gray", vmin=0, vmax=255)
    ax.contour(frame["mask"], levels=[.5], colors=[CYAN], linestyles="--", linewidths=1.1)
    if mask.any():
        ax.contour(mask, levels=[.5], colors=[GOLD], linewidths=1.1)
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()


def generate(prepared, predictions, results, expert, training, output):
    prepared, predictions, results, expert, training, output = map(Path, (prepared, predictions, results, expert, training, output))
    meta = verify_results(results)
    verify_results(expert)
    if sha256(predictions / "run_metadata.json") != meta["prediction_metadata_sha256"]:
        raise ValueError("Figures require the predictions used in evaluation")
    summary = json.loads((results / "summary.json").read_text())
    patients, frames = read_csv(results / "patients.csv"), read_csv(results / "frames.csv")
    baseline = read_csv(expert / "patients.csv")
    data = CamusDataset(prepared, "test")
    prediction_index = {r["frame_id"]: r for r in read_csv(predictions / "predictions.csv")}
    frame_index = {r["frame_id"]: r for r in frames}
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "svg.hashsalt": "camus523"})
    files = []

    def save(fig, name):
        for extension in ("png", "svg"):
            filename = f"{name}.{extension}"
            fig.savefig(output / filename, dpi=170, metadata={"Date": None} if extension == "svg" else None)
            files.append(filename)
        plt.close(fig)

    def get_mask(frame):
        record = prediction_index[frame["frame_id"]]
        path = predictions / record["path"]
        if sha256(path) != record["sha256"]:
            raise ValueError("Prediction array changed before plotting")
        with np.load(path, allow_pickle=False) as values:
            return values["mask"]

    reference = np.array([float(r["camus_reference_ef_pct"]) for r in patients])
    estimated = np.array([float(r["predicted_ef_pct"]) for r in patients])
    median_dice = np.median([float(r["mean_frame_dice"]) for r in patients])
    example = min(patients, key=lambda r: (abs(float(r["mean_frame_dice"])-median_dice), r["patient_id"]))
    frame = data.patient_frames(example["patient_id"])["2CH_ED"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    identity_scatter(axes[0], [float(r["reference_ef_pct"]) for r in baseline],
                     [float(r["ef_pct"]) for r in baseline], "A  Expert-mask geometry\n50/50 exact matches; MAE 0.00 pp", "Reproduced EF (%)")
    overlay(axes[1], frame, get_mask(frame), "B  GroupNorm U-Net\nNative LV cavity contours")
    axes[1].legend(handles=[Line2D([], [], color=CYAN, linestyle="--", label="Expert"),
                            Line2D([], [], color=GOLD, label="Predicted")], loc="lower center", fontsize=8, ncol=2)
    identity_scatter(axes[2], reference, estimated, "C  Integrated image-to-EF pipeline\n50 test patients; MAE 4.96 pp")
    save(fig, "overview")

    history = read_csv(training / "history.csv")
    trained = json.loads((training / "run_metadata.json").read_text())
    epoch = [int(r["epoch"]) for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    axes[0].plot(epoch, [float(r["validation_dice"]) for r in history], color=BLUE, label="GroupNorm U-Net")
    axes[0].scatter([trained["best_epoch"]], [trained["best_validation_dice"]], color=GOLD, marker="*", s=110,
                    zorder=4, label=f"Selected epoch {trained['best_epoch']}")
    axes[0].set(xlabel="Epoch", ylabel="Validation Dice (model grid)", ylim=(.7, 1), title="Checkpoint selection")
    axes[0].legend(fontsize=8)
    for field, color, label in (("train_loss", BLUE, "Augmented training"), ("validation_loss", GOLD, "Validation")):
        axes[1].plot(epoch, [float(r[field]) for r in history], color=color, label=label)
    axes[1].set(xlabel="Epoch", ylabel="0.5 BCE + 0.5 soft Dice loss", title="Optimization")
    axes[1].legend(fontsize=8)
    for ax in axes: style(ax)
    save(fig, "training_history")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    identity_scatter(axes[0], reference, estimated, "EF agreement | n = 50")
    error, mean = estimated-reference, (estimated+reference)/2
    stats = agreement(error)
    axes[1].scatter(mean, error, s=27, alpha=.8, color=BLUE, edgecolors="none")
    axes[1].set(xlabel="Mean of estimated and reference EF (%)", ylabel="Estimated − reference EF (pp)",
                xlim=(0, 100), ylim=(-35, 25), title="Bland–Altman analysis")
    for key, label, linestyle in (("bias", "Bias", "-"), ("loa_lower", "Lower LoA", "--"), ("loa_upper", "Upper LoA", "--")):
        axes[1].axhline(stats[key], color=".4", linewidth=1, linestyle=linestyle)
        axes[1].text(99, stats[key]+.4, f"{label}: {stats[key]:.2f}", ha="right", fontsize=8,
                     bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8, "pad": .5})
    style(axes[1])
    save(fig, "ef_agreement")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    rng = np.random.default_rng(523)
    for ax, field, label in zip(axes, ("dice", "hd95_mm"), ("Native-grid Dice", "HD95 (mm)")):
        values = [[float(r[field]) for r in frames if r["view_phase"] == group] for group in VIEWS]
        ax.boxplot(values, labels=[v.replace("_", " ") for v in VIEWS], showfliers=False,
                   medianprops={"color": GOLD, "linewidth": 1.7})
        for i, group in enumerate(values, 1):
            ax.scatter(i+rng.uniform(-.14, .14, len(group)), group, s=12, color=BLUE, alpha=.45, edgecolors="none")
        ax.set(ylabel=label, title="50 patients per view and phase")
        if field == "dice": ax.set_ylim(.7, 1.01)
        else: ax.set_ylim(bottom=0)
        style(ax)
    save(fig, "segmentation_metrics")

    representatives = []
    for quality in ("Good", "Medium", "Poor"):
        rows = [r for r in patients if r["image_quality"] == quality]
        median = np.median([abs(float(r["ef_error_vs_camus_pp"])) for r in rows])
        representatives.append(min(rows, key=lambda r: (abs(abs(float(r["ef_error_vs_camus_pp"]))-median), r["patient_id"])))
    worst = sorted(patients, key=lambda r: (-abs(float(r["ef_error_vs_camus_pp"])), r["patient_id"]))[:3]
    for name, rows, title in (("prediction_examples", representatives, "Representative patients by image quality"),
                              ("largest_ef_errors", worst, "Patients with the largest absolute EF errors")):
        fig, axes = plt.subplots(3, 4, figsize=(12, 10.2), constrained_layout=True)
        for i, row in enumerate(rows):
            native = data.patient_frames(row["patient_id"])
            for j, key in enumerate(VIEWS):
                frame = native[key]
                score = float(frame_index[frame["frame_id"]]["dice"])
                label = f"{key.replace('_', ' ')} | Dice {score:.3f}"
                if j == 0:
                    label = f"{row['patient_id']} | {row['image_quality']}\nEF error {float(row['ef_error_vs_camus_pp']):+.0f} pp\n"+label
                overlay(axes[i, j], frame, get_mask(frame), label)
        fig.suptitle(title+"\nExpert: cyan dashed; predicted: gold solid", fontsize=12)
        save(fig, name)
    atomic_json(output / "figure_manifest.json", {
        "analysis_metadata_sha256": sha256(results / "run_metadata.json"),
        "expert_metadata_sha256": sha256(expert / "run_metadata.json"),
        "plotting_code_sha256": sha256(Path(__file__)),
        "overview_patient": example["patient_id"],
        "representative_patients": [r["patient_id"] for r in representatives],
        "largest_error_patients": [r["patient_id"] for r in worst],
        "selection": "Closest to within-quality median absolute EF error; patient ID breaks ties. Overview: closest to median patient-mean Dice.",
        "artifact_sha256": {name: sha256(output / name) for name in files}})
    print(f"Generated {len(files)//2} figures in PNG and SVG.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared"))
    parser.add_argument("--predictions", type=Path, default=Path("outputs/predictions"))
    parser.add_argument("--results", type=Path, default=Path("outputs/analysis"))
    parser.add_argument("--expert", type=Path, default=Path("outputs/geometry"))
    parser.add_argument("--training", type=Path, default=Path("segmentation/model_artifact"))
    parser.add_argument("--output", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()
    generate(args.prepared, args.predictions, args.results, args.expert, args.training, args.output)
