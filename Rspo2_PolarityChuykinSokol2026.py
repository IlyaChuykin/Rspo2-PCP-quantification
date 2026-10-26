#!/usr/bin/env python3
"""
Rspo2 polarity quantification, revised clean version.

Main improvements over the older script:
- selected-cell outlines are baked into the PNG as solid bright yellow pixels
  (not faint matplotlib contour lines)
- separate PNGs are created with one polarity vector per selected cell
- vectors are direction-only display arrows, normalized to cell size for visibility
- anterior (top) and posterior (bottom) labels are added to vector PNGs

Angle convention:
  0 deg   = up = anterior
  90 deg  = right
  180 deg = down = posterior
  270 deg = left
"""

from pathlib import Path
import argparse, math
import numpy as np
import pandas as pd
from PIL import Image, ImageSequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from skimage.morphology import erosion, dilation, disk
from skimage.segmentation import find_boundaries


def read_mask(path):
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == ():
        obj = obj.item()
    if isinstance(obj, dict):
        return np.asarray(obj["masks"]).astype(np.int32)
    return np.asarray(obj).astype(np.int32)


def read_signal(path):
    im = Image.open(path)
    frames = []
    for fr in ImageSequence.Iterator(im):
        arr = np.asarray(fr)
        if arr.ndim == 3:
            arr = arr[..., 0]
        frames.append(arr.astype(float))
    return frames[0] if len(frames) == 1 else np.max(np.stack(frames, axis=0), axis=0)


def membrane_band(cell_mask, width_px=2):
    er = erosion(cell_mask, disk(width_px))
    band = cell_mask & (~er)
    return band if band.sum() >= 10 else cell_mask


def grayscale_rgb(signal, p_low=1, p_high=99.5):
    lo = float(np.percentile(signal, p_low))
    hi = float(np.percentile(signal, p_high))
    if hi <= lo:
        hi = lo + 1.0
    x = np.clip((signal - lo) / (hi - lo), 0, 1)
    g = np.round(x * 255).astype(np.uint8)
    return np.dstack([g, g, g])


def make_bright_outline_rgb(signal, selected_mask, thickness=2):
    """Return an RGB image with opaque bright yellow selected-cell boundaries."""
    rgb = grayscale_rgb(signal)
    boundary = find_boundaries(selected_mask, mode="outer")
    # Dilate the boundary so the saved line is clearly visible.
    boundary = dilation(boundary, disk(max(1, int(thickness))))
    rgb[boundary] = np.array([255, 255, 0], dtype=np.uint8)  # pure yellow
    return rgb


def add_orientation_labels(ax):
    effects = [pe.withStroke(linewidth=3, foreground="black")]
    ax.text(0.5, 0.985, "ANTERIOR", transform=ax.transAxes,
            ha="center", va="top", fontsize=14, fontweight="bold",
            color="white", path_effects=effects)
    ax.annotate("", xy=(0.5, 0.94), xytext=(0.5, 0.86),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=3.0, color="white"))

    ax.text(0.5, 0.015, "POSTERIOR", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=14, fontweight="bold",
            color="white", path_effects=effects)
    ax.annotate("", xy=(0.5, 0.06), xytext=(0.5, 0.14),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=3.0, color="white"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", default="E2Phallmask_seg.npy",
                        help="Cellpose mask filename. Default: E2Phallmask_seg.npy")
    parser.add_argument("--signal", default="E2Rspo2.tif",
                        help="Rspo2 TIFF filename. Default: E2Rspo2.tif")
    parser.add_argument("--outdir", default="E2_Rspo2_quantification",
                        help="Output folder name. Default: E2_Rspo2_quantification")
    parser.add_argument("--cutoffs", nargs="+", type=float, default=[60, 40, 20])
    parser.add_argument("--outline-thickness", type=int, default=2,
                        help="Boundary dilation radius in pixels; 2 is thick and bright")
    parser.add_argument("--vector-scale", type=float, default=0.75,
                        help="Displayed arrow length as a fraction of equivalent cell radius")
    args = parser.parse_args()

    # Resolve all relative paths from the folder containing this script.
    # This means you can simply place the script in the Quantification folder
    # next to E2Phallmask_seg.npy and E2Rspo2.tif, then run it without arguments.
    script_dir = Path(__file__).resolve().parent
    mask_path = Path(args.mask)
    signal_path = Path(args.signal)
    outdir = Path(args.outdir)
    if not mask_path.is_absolute():
        mask_path = script_dir / mask_path
    if not signal_path.is_absolute():
        signal_path = script_dir / signal_path
    if not outdir.is_absolute():
        outdir = script_dir / outdir

    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    if not signal_path.exists():
        raise FileNotFoundError(f"Rspo2 signal file not found: {signal_path}")

    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Mask:   {mask_path.name}")
    print(f"Signal: {signal_path.name}")
    print(f"Output: {outdir}")

    mask = read_mask(mask_path)
    signal = read_signal(signal_path)
    if mask.shape != signal.shape:
        raise ValueError(f"Mask and signal dimensions do not match: {mask.shape} vs {signal.shape}")

    labels = [int(x) for x in np.unique(mask) if int(x) != 0]
    rows = []

    for lab in labels:
        cell = mask == lab
        ys, xs = np.nonzero(cell)
        band = membrane_band(cell, 2)
        yb, xb = np.nonzero(band)
        vals = signal[band].astype(float)

        cx, cy = xs.mean(), ys.mean()
        area = int(cell.sum())
        eq = math.sqrt(area / math.pi)
        wsum = vals.sum()

        if wsum > 0:
            sx = (xb * vals).sum() / wsum
            sy = (yb * vals).sum() / wsum
        else:
            sx = xb.mean()
            sy = yb.mean()

        dx = sx - cx
        dy_img = sy - cy
        dy_up = cy - sy
        angle = (math.degrees(math.atan2(dx, dy_up)) + 360) % 360
        strength = math.sqrt(dx*dx + dy_up*dy_up) / eq if eq > 0 else np.nan

        rows.append({
            "cell_id": lab,
            "centroid_x": float(cx),
            "centroid_y": float(cy),
            "area_px": area,
            "equivalent_radius_px": float(eq),
            "mean_membrane_intensity": float(vals.mean()),
            "p95_membrane_intensity": float(np.percentile(vals, 95)),
            "signal_centroid_x": float(sx),
            "signal_centroid_y": float(sy),
            "vector_x_right_positive": float(dx),
            "vector_y_image_down_positive": float(dy_img),
            "polarity_strength": float(strength),
            "angle_deg_0_is_up": float(angle),
            "alignment_to_anterior": float(np.cos(np.deg2rad(angle))),
        })

    cells = pd.DataFrame(rows)
    n_bg = max(1, int(np.ceil(0.20 * len(cells))))
    background = float(cells.nsmallest(n_bg, "mean_membrane_intensity")["mean_membrane_intensity"].mean())

    bin_edges = np.arange(0, 360 + 22.5, 22.5)
    summary_rows = []

    for cutoff in args.cutoffs:
        cutoff_i = int(cutoff)
        threshold = background * (1 + cutoff / 100.0)
        col = f"positive_background{cutoff_i}"
        cells[col] = cells["mean_membrane_intensity"] >= threshold
        pos = cells[cells[col]].copy()
        pos.to_csv(outdir / f"Rspo2_candidate_positive_background{cutoff_i}.csv", index=False)

        counts, _ = np.histogram(pos["angle_deg_0_is_up"].to_numpy(float), bins=bin_edges)
        pd.DataFrame({
            "bin_start_deg": bin_edges[:-1],
            "bin_end_deg": bin_edges[1:],
            "n_cells": counts
        }).to_csv(outdir / f"Rspo2_rose_background{cutoff_i}_raw_counts_by_bin.csv", index=False)

        summary_rows.append({
            "cutoff_name": f"background+{cutoff_i}%",
            "background_estimate_lowest20pct_cells": background,
            "threshold_intensity": threshold,
            "n_positive_cells": int(len(pos)),
            "n_total_cells": int(len(cells)),
            "mean_alignment_to_anterior": float(pos["alignment_to_anterior"].mean()) if len(pos) else np.nan,
            "median_alignment_to_anterior": float(pos["alignment_to_anterior"].median()) if len(pos) else np.nan,
            "fraction_positive_cells_aligned_anterior": float((pos["alignment_to_anterior"] > 0).mean()) if len(pos) else np.nan,
            "mean_polarity_strength": float(pos["polarity_strength"].mean()) if len(pos) else np.nan,
        })

        selected_ids = pos["cell_id"].astype(int).to_numpy()
        selected_mask = np.where(np.isin(mask, selected_ids), mask, 0)
        overlay_rgb = make_bright_outline_rgb(signal, selected_mask, args.outline_thickness)

        # 1) BRIGHT OUTLINE PNG
        fig, ax = plt.subplots(figsize=(8, 12))
        ax.imshow(overlay_rgb)
        for _, r in pos.iterrows():
            ax.text(r["centroid_x"], r["centroid_y"], str(int(r["cell_id"])),
                    fontsize=5.2, color="white", ha="center", va="center",
                    path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])
        ax.set_title(
            f"Selected Rspo2-positive cells: background +{cutoff_i}%\n"
            f"threshold={threshold:.3f}, n={len(pos)}/{len(cells)}"
        )
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(outdir / f"Rspo2_positive_cells_background{cutoff_i}_BRIGHT_outline.png", dpi=220)
        plt.close(fig)

        # 2) VECTOR PNG: same bright outlines + one direction arrow per selected cell
        fig, ax = plt.subplots(figsize=(8, 12))
        ax.imshow(overlay_rgb)

        for _, r in pos.iterrows():
            angle = math.radians(float(r["angle_deg_0_is_up"]))
            # Convert angle convention to image coordinates: +x right, +y down.
            ux = math.sin(angle)
            uy = -math.cos(angle)
            L = max(8.0, args.vector_scale * float(r["equivalent_radius_px"]))
            x0 = float(r["centroid_x"])
            y0 = float(r["centroid_y"])
            x1 = x0 + ux * L
            y1 = y0 + uy * L
            ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>", color="magenta", lw=2.2,
                    mutation_scale=12, shrinkA=0, shrinkB=0
                )
            )

        add_orientation_labels(ax)
        ax.set_title(
            f"Rspo2 polarity vectors: background +{cutoff_i}%\n"
            f"0° = anterior (up), 180° = posterior (down), n={len(pos)}"
        )
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(outdir / f"Rspo2_vectors_background{cutoff_i}.png", dpi=220)
        plt.close(fig)

        # 3) Rose plot
        centers = np.deg2rad((bin_edges[:-1] + bin_edges[1:]) / 2)
        widths = np.deg2rad(np.diff(bin_edges))
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="polar")
        ax.bar(centers, counts, width=widths, align="center", alpha=0.9)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title(f"Rspo2 polarity rose plot, background +{cutoff_i}%\nraw counts, n={len(pos)}")
        fig.tight_layout()
        fig.savefig(outdir / f"Rspo2_rose_background{cutoff_i}_raw_counts.png", dpi=220)
        plt.close(fig)

    cells.to_csv(outdir / "Rspo2_all_cells_background_cutoff_classification.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(outdir / "Rspo2_summary_background_cutoffs.csv", index=False)
    print("Quantification finished successfully.")


if __name__ == "__main__":
    main()
