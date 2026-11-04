"""Export scientific figures from the verified controlled-error experiment."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.ndimage import binary_fill_holes

from data.dataset import CamusDataset
from data.download_camus import atomic_json, sha256
from data.protocol import identity, verify_results
from geometry.comparators import measure_mask
from .perturbations import FRAME_KEYS, METHODS, perturb_mask
from .run import read_csv

METHOD_NAMES = {"publisher_original": "CAMUS publisher", "simpson_biplane": "Phase-specific Simpson",
                "area_length_biplane_ellipsoid": "Biplane area-length"}
BLUE, GOLD, PURPLE = "#24658A", "#D89020", "#74648B"
COLORS = {"2CH": BLUE, "4CH": GOLD, "both": PURPLE}
FEATURE_LABELS = {"area_error_pct": "Area error (%)", "length_error_mm": "Long-axis error (mm)",
                  "abs_area_error_pct": "Absolute area error (%)", "abs_length_error_mm": "Absolute axis error (mm)",
                  "apex_displacement_mm": "Apex displacement (mm)",
                  "base_midpoint_displacement_mm": "Basal-midpoint displacement (mm)", "one_minus_dice": "1 - Dice"}


def number(row, field):
    return float(row[field]) if row[field] not in (None, "") else np.nan


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.15)
    ax.set_axisbelow(True)


def draw_native(ax, frame, mask, original_geometry, changed_geometry):
    sx, sy = frame["geometry"]["spacing_xy_mm"]
    h, w = frame["mask"].shape
    ax.imshow(frame["image"], cmap="gray", vmin=0, vmax=255,
              extent=(-.5*sx, (w-.5)*sx, (h-.5)*sy, -.5*sy))
    ax.contour(np.arange(w)*sx, np.arange(h)*sy, frame["mask"], levels=[.5], colors=[BLUE], linestyles="--", linewidths=1.3)
    ax.contour(np.arange(w)*sx, np.arange(h)*sy, mask, levels=[.5], colors=[GOLD], linewidths=1.1)
    for geometry, color, line in ((original_geometry, BLUE, "--"), (changed_geometry, GOLD, "-")):
        if geometry:
            base, apex = np.array(geometry["base_midpoint_xy_mm"]), np.array(geometry["apex_xy_mm"])
            ax.plot([base[0], apex[0]], [base[1], apex[1]], line, color=color, alpha=.8, linewidth=.9)
            ax.plot(base[0], base[1], "s", color=color, markersize=3)
            ax.plot(apex[0], apex[1], "o", color=color, markersize=3)
    yy, xx = np.nonzero(frame["mask"])
    ax.set(xlim=(xx.min()*sx-8, xx.max()*sx+8), ylim=(yy.max()*sy+8, yy.min()*sy-8), aspect="equal")
    ax.tick_params(labelsize=7)


def generate(prepared, predictions, results, output):
    prepared, predictions, results, output = map(Path, (prepared, predictions, results, output))
    meta = verify_results(results)
    if meta["data_identity"] != identity(prepared) or meta["prediction_metadata_sha256"] != sha256(predictions/"run_metadata.json"):
        raise ValueError("Figure inputs differ from the experiment")
    summary = read_csv(results/"sensitivity_summary.csv")
    observed = read_csv(results/"observed_patients.csv")
    frames = read_csv(results/"observed_frames.csv")
    audits = read_csv(results/"perturbation_frames.csv")
    associations = read_csv(results/"associations.csv")
    dataset = CamusDataset(prepared, "test")
    prediction_index = {r["frame_id"]: r for r in read_csv(predictions/"predictions.csv")}
    primary = [r for r in observed if r["method"] == "publisher_original"]
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
                         "svg.hashsalt": "camus523-sensitivity", "svg.fonttype": "none"})
    files = []

    def save(fig, name):
        for extension in ("png", "svg"):
            filename = f"{name}.{extension}"
            fig.savefig(output/filename, dpi=180, metadata={"Date": None} if extension == "svg" else None)
            files.append(filename)
        plt.close(fig)

    def sensitivity(method, family, magnitude, view, phase, field="ef_pp_mean"):
        matches = [r for r in summary if r["method"] == method and r["group_type"] == "overall"
                   and ((magnitude == 0 and r["family"] == "control") or
                        (r["family"] == family and float(r["magnitude_mm"]) == magnitude
                         and r["view_scope"] == view and r["phase_scope"] == phase))]
        if len(matches) != 1:
            raise ValueError("Figure selection is not unique")
        return matches[0]

    median = np.median([float(r["mean_frame_dice"]) for r in primary])
    representative = min(primary, key=lambda r: (abs(float(r["mean_frame_dice"])-median), r["patient_id"]))["patient_id"]
    native = dataset.patient_frames(representative)
    interventions = [("apical_truncation", 2., "Apical cut: 2 mm"),
                     ("basal_displacement", 2., "Basal inward: 2 mm"),
                     ("basal_displacement", -2., "Basal outward: 2 mm")]
    fig, axes = plt.subplots(3, 4, figsize=(12.5, 10), constrained_layout=True)
    for j, key in enumerate(FRAME_KEYS):
        frame = native[key]
        geometry = measure_mask(frame["mask"], frame["geometry"]["spacing_xy_mm"])
        for i, (family, d, label) in enumerate(interventions):
            changed, _ = perturb_mask(frame["mask"], frame["geometry"]["spacing_xy_mm"], geometry, family, d)
            try:
                gm = measure_mask(changed, frame["geometry"]["spacing_xy_mm"])
            except ValueError:
                gm = None
            draw_native(axes[i,j], frame, changed, geometry, gm)
            axes[i,j].set_title(key.replace("_", " ") if i == 0 else "", fontsize=10)
            if j == 0:
                axes[i,j].set_ylabel(label+"\ny (mm)")
            if i == 2:
                axes[i,j].set_xlabel("x (mm)")
    fig.suptitle(f"Controlled native-mask interventions | {representative}\nBlue dashed: expert; gold solid: perturbed. Circles: apex; squares: basal midpoint.", fontsize=12)
    save(fig, "perturbation_overlays")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), sharey=True, constrained_layout=True)
    for i, (family, values) in enumerate((("apical_truncation", [0,1,2]), ("basal_displacement", [-2,-1,0,1,2]))):
        for j, phase in enumerate(("ED", "ES", "both")):
            ax = axes[i,j]
            for view, marker in (("2CH", "o"), ("4CH", "s"), ("both", "^")):
                rows = [sensitivity("publisher_original", family, d, view, phase) for d in values]
                means = [number(r, "ef_pp_mean") for r in rows]
                low = [number(r, "ef_pp_mean_ci95_low") for r in rows]
                high = [number(r, "ef_pp_mean_ci95_high") for r in rows]
                ax.plot(values, means, marker=marker, color=COLORS[view], label=view if view != "both" else "Both views", linewidth=1.4, markersize=4)
                ax.fill_between(values, low, high, color=COLORS[view], alpha=.12)
            ax.axhline(0, color=".4", linewidth=.8, linestyle="--")
            ax.set_title(("Apical cut" if i == 0 else "Basal displacement")+" | "+("Both phases" if phase == "both" else phase+" only"))
            ax.set_xticks(values)
            ax.set_xlabel("Truncation (mm)" if i == 0 else "Displacement (mm; positive = inward)")
            if j == 0:
                ax.set_ylabel("Mean signed EF change (pp)")
            style(ax)
    axes[0,0].legend(fontsize=8)
    fig.suptitle("Publisher calculation: paired sensitivity in 50 test patients\nShading: 95% patient-bootstrap intervals; lines connect tested magnitudes.", fontsize=12)
    save(fig, "sensitivity_curves")

    levels = [("apical_truncation", 1), ("apical_truncation", 2)]+[("basal_displacement", d) for d in (-2,-1,1,2)]
    scopes = [(v,p) for v in ("2CH","4CH","both") for p in ("ED","ES","both")]
    arrays = [np.array([[number(sensitivity(m,f,d,v,p),"ef_pp_mae") for v,p in scopes] for f,d in levels]) for m in METHODS]
    vmax = max(np.nanmax(a) for a in arrays)
    fig, axes = plt.subplots(3, 1, figsize=(11, 10.5), constrained_layout=True)
    for ax, method, values in zip(axes, METHODS, arrays):
        im = ax.imshow(values, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(METHOD_NAMES[method])
        ax.set_xticks(range(9), [f"{v if v != 'both' else 'Both views'}\n{p if p != 'both' else 'ED + ES'}" for v,p in scopes], fontsize=8)
        ax.set_yticks(range(6), [f"Apical cut {d:g} mm" if f == "apical_truncation" else f"Basal {d:+g} mm" for f,d in levels], fontsize=9)
        for y,x in np.ndindex(values.shape):
            value=values[y,x]
            family, magnitude = levels[y]
            view, phase = scopes[x]
            count = int(sensitivity(method,family,magnitude,view,phase)["ef_pp_numeric_pairs"])
            label = f"{value:.2f}" if np.isfinite(value) else "NA"
            if count < 50:
                label += f"\nn={count}"
            ax.text(x,y,label,ha="center",va="center",fontsize=8,
                    color="white" if value > .6*vmax else "#202020")
    fig.colorbar(im,ax=axes,label="Mean absolute EF change (pp)",shrink=.7,pad=.02)
    fig.suptitle("Geometric-method sensitivity | 50 attempted patients per cell\nEach change is relative to that method's unperturbed expert-mask estimate.",fontsize=12)
    save(fig,"method_sensitivity")

    features = list(FEATURE_LABELS)
    overall_associations = [r for r in associations if r["group_type"] == "overall"]
    fig, axes = plt.subplots(1,3,figsize=(13.5,5.2),sharey=True,constrained_layout=True)
    for ax,method in zip(axes,METHODS):
        values = np.array([[number(next(r for r in overall_associations if r["method"]==method and r["view_phase"]==vp and r["feature"]==f),"spearman_rho") for vp in FRAME_KEYS] for f in features])
        im=ax.imshow(values,vmin=-1,vmax=1,cmap="RdBu_r",aspect="auto")
        ax.set_title(METHOD_NAMES[method])
        ax.set_xticks(range(4),[k.replace("_","\n") for k in FRAME_KEYS],fontsize=9)
        ax.set_yticks(range(len(features)),[FEATURE_LABELS[f] for f in features],fontsize=9)
        for y,x in np.ndindex(values.shape):
            value=values[y,x]
            ax.text(x,y,f"{value:.2f}" if np.isfinite(value) else "NA",ha="center",va="center",fontsize=9,color="white" if abs(value)>.65 else "#202020")
    fig.colorbar(im,ax=axes,label="Spearman rho",shrink=.8,pad=.02)
    fig.suptitle("Observed contour errors and EF error | one frame per patient in each column\nFirst two rows: signed EF error; remaining rows: absolute EF error. Exploratory associations.",fontsize=12)
    save(fig,"association_matrix")

    by_patient={r["patient_id"]:r for r in primary}
    subset=[r for r in frames if r["view_phase"]=="2CH_ES"]
    fig,axes=plt.subplots(2,2,figsize=(10.5,8),constrained_layout=True)
    chosen=["area_error_pct","length_error_mm","apex_displacement_mm","base_midpoint_displacement_mm"]
    for ax,feature in zip(axes.flat,chosen):
        signed=feature in ("area_error_pct","length_error_mm")
        pairs=[(number(r,feature),number(by_patient[r["patient_id"]],"ef_error_vs_gt_pp")) for r in subset]
        x=np.array([a for a,b in pairs]); y=np.array([b if signed else abs(b) for a,b in pairs])
        valid=np.isfinite(x)&np.isfinite(y)
        ax.scatter(x[valid],y[valid],s=27,color=BLUE,alpha=.8,edgecolors="white",linewidths=.4)
        association=next(r for r in overall_associations if r["method"]=="publisher_original" and r["view_phase"]=="2CH_ES" and r["feature"]==feature)
        ax.set_title(f"n = {association['n']}; rho = {number(association,'spearman_rho'):.2f}\n95% CI {number(association,'ci95_low'):.2f} to {number(association,'ci95_high'):.2f}")
        ax.set_xlabel(FEATURE_LABELS[feature])
        ax.set_ylabel(("Signed" if signed else "Absolute")+" EF error (pp)")
        ax.axhline(0,color=".5",linestyle="--",linewidth=.7)
        style(ax)
    fig.suptitle("Observed 2CH ES errors | publisher EF calculation\nThis phase had the lowest Dice in the completed baseline analysis.",fontsize=12)
    save(fig,"observed_error_associations")

    outliers=sorted(primary,key=lambda r:(-abs(float(r["ef_error_vs_gt_pp"])),r["patient_id"]))[:3]
    fig,axes=plt.subplots(3,5,figsize=(16,9.3),constrained_layout=True,gridspec_kw={"width_ratios":[1,1,1,1,1.15]})
    for i,patient in enumerate(outliers):
        native=dataset.patient_frames(patient["patient_id"])
        for j,key in enumerate(FRAME_KEYS):
            frame=native[key]; row=next(r for r in frames if r["frame_id"]==frame["frame_id"])
            record=prediction_index[frame["frame_id"]]; path=predictions/record["path"]
            if sha256(path)!=record["sha256"]:
                raise ValueError("Prediction changed before plotting")
            with np.load(path,allow_pickle=False) as a:
                mask=a["mask"]
            geometries=[]
            for prefix in ("gt","measured"):
                geometries.append({point+"_xy_mm":[number(row,f"{prefix}_{point}_{dim}_mm") for dim in ("x","y")]
                                   for point in ("apex","base_midpoint")})
            draw_native(axes[i,j],frame,mask,*geometries)
            axes[i,j].set_title(key.replace("_"," ") if i==0 else "",fontsize=10)
            if j==0:
                axes[i,j].set_ylabel(f"{patient['patient_id']}\nEF error {float(patient['ef_error_vs_gt_pp']):+.0f} pp")
        ax=axes[i,4]
        values=[float(patient[k]) for k in ("edv_contribution_pp","esv_contribution_pp","ef_rounding_remainder_pp")]
        ax.barh([2,1,0],values,color=[BLUE,GOLD,"#777777"],height=.55)
        ax.set_yticks([2,1,0],["EDV term","ESV term","Rounding"],fontsize=8)
        for y,value in zip([2,1,0],values):
            ax.text(value,y,f" {value:+.1f}" if value>=0 else f"{value:+.1f} ",ha="left" if value>=0 else "right",va="center",fontsize=8)
        ax.set_xlim(-32,22)
        ax.axvline(0,color=".3",linewidth=.7)
        ax.set_xlabel("EF contribution (pp)")
        ax.set_title("Arithmetic decomposition" if i==0 else "",fontsize=10)
        style(ax)
    fig.suptitle("Largest original EF errors | expert (blue dashed) and predicted (gold solid)\nLandmarks use the independent mask detector; contribution terms sum to the publisher EF error.",fontsize=12)
    save(fig,"largest_error_anatomy")

    failures=[r for r in audits if r["geometry_status"]!="ok"]
    if failures:
        fig,axes=plt.subplots(len(failures),2,figsize=(9,4*len(failures)),squeeze=False,constrained_layout=True)
        for i,row in enumerate(failures):
            frame=dataset.patient_frames(row["patient_id"])[row["view_phase"]]
            spacing=frame["geometry"]["spacing_xy_mm"]
            gm=measure_mask(frame["mask"],spacing)
            changed,_=perturb_mask(frame["mask"],spacing,gm,row["family"],float(row["magnitude_mm"]))
            holes=binary_fill_holes(changed.astype(bool)) & ~changed.astype(bool)
            draw_native(axes[i,0],frame,changed,gm,None)
            axes[i,0].set_title(row["frame_id"]+"\nBasal outward 2 mm",fontsize=10)
            yy,xx=np.nonzero(holes)
            if len(yy):
                cy,cx=int(yy[0]),int(xx[0]); radius=6
                zoom=changed[max(0,cy-radius):cy+radius+1,max(0,cx-radius):cx+radius+1]
                axes[i,1].imshow(zoom,cmap="gray",vmin=0,vmax=1,interpolation="nearest")
                axes[i,1].plot(cx-max(0,cx-radius),cy-max(0,cy-radius),"s",mfc="none",mec=GOLD,markersize=24,markeredgewidth=2)
                axes[i,0].plot(cx*spacing[0],cy*spacing[1],"s",mfc="none",mec=GOLD,markersize=12,markeredgewidth=1.5)
                axes[i,1].set_title(f"Raster close-up: {len(yy)} interior background pixel(s)",fontsize=10)
            axes[i,1].set_axis_off()
        fig.suptitle("Secondary-method failures after native-grid resampling\nThe fixed no-cleanup protocol retains these masks and reports their rejection.",fontsize=12)
        save(fig,"rasterization_failures")

    atomic_json(output/"figure_manifest.json",{"source_results_metadata_sha256":sha256(results/"run_metadata.json"),
        "source_artifact_sha256":meta["artifact_sha256"],"plotting_code_sha256":sha256(Path(__file__)),
        "representative_patient":representative,"representative_selection":"Closest to median patient-mean Dice; tie by patient ID",
        "outlier_patients":[r["patient_id"] for r in outliers],"outlier_selection":"Three largest original absolute EF errors; tie by patient ID",
        "failed_frame_variants":[r["frame_id"]+":"+r["family"]+":"+r["magnitude_mm"] for r in failures],
        "artifact_sha256":{name:sha256(output/name) for name in files}})
    print(json.dumps({"figures":len(files)//2,"representative":representative,"outliers":[r["patient_id"] for r in outliers]},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared",type=Path,default=Path("data/prepared"))
    parser.add_argument("--predictions",type=Path,default=Path("outputs/predictions"))
    parser.add_argument("--results",type=Path,default=Path("results/sensitivity_v1"))
    parser.add_argument("--output",type=Path,default=Path("results/sensitivity_v1/figures"))
    args=parser.parse_args()
    generate(args.prepared,args.predictions,args.results,args.output)
