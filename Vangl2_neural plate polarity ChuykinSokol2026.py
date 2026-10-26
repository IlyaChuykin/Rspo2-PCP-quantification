#!/usr/bin/env python3
"""
Reusable Vangl2 polarity quantification for one Cellpose-segmented image.

IMPORTANT:
- The Cellpose mask is treated as definitive.
- Cells manually deleted in Cellpose are excluded.
- Cells kept in the saved mask are quantified, even if close to an image edge.
- There is NO additional automatic edge-cell exclusion in Python.

Angle convention:
    0 deg   = anterior / top
    90 deg  = right
    180 deg = posterior / bottom
    -90 deg = left

Method preserved from the previous analysis:
- 3-pixel inner membrane ring
- minimum cell area = 200 pixels
- minimum ring size = 40 pixels
- polarity weights = Vangl2 intensity above that cell's ring mean
- polarity vector = weighted mean of unit radial directions around the cell perimeter

Outputs:
- bright yellow mask overlay
- yellow mask overlay + individual polarity vectors
- per-cell CSV
- summary CSV
- rose plot
"""

from pathlib import Path
import argparse
import csv
import math

import numpy as np
import tifffile as tiff

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# DEFAULT ANALYSIS SETTINGS
# =========================

RING_WIDTH_PIXELS = 3
MIN_CELL_AREA_PIXELS = 200
MIN_RING_PIXELS = 40

LOWER_PERCENTILE = 1
UPPER_PERCENTILE = 99.5

# Display only; does not affect the measured vector.
ARROW_LENGTH_PIXELS = 12

# Bright yellow mask boundaries.
BOUNDARY_RGB = np.array([1.0, 1.0, 0.0])
BOUNDARY_THICKNESS = 2

# Vector display color.
ARROW_COLOR = "magenta"


# =========================
# I/O
# =========================

def load_cellpose_masks(path):
    data = np.load(path, allow_pickle=True)

    if isinstance(data, np.ndarray) and data.shape == ():
        data = data.item()

    if isinstance(data, dict):
        if "masks" not in data:
            raise ValueError(f"{path.name} does not contain a 'masks' entry.")
        masks = np.asarray(data["masks"])
    else:
        masks = np.asarray(data)

    if masks.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {masks.shape}")

    return masks.astype(np.int32)


def load_image_2d(path, channel="red"):
    img = tiff.imread(path)
    img = np.squeeze(img)

    print(path.name, "raw shape:", img.shape)

    # RGB / RGBA stored as Y, X, C
    if img.ndim == 3 and img.shape[-1] in (3, 4):
        if channel == "red":
            img = img[..., 0]
        elif channel == "green":
            img = img[..., 1]
        elif channel == "blue":
            img = img[..., 2]
        else:
            img = img[..., :3].max(axis=-1)

    # RGB / RGBA stored as C, Y, X
    elif img.ndim == 3 and img.shape[0] in (3, 4):
        if channel == "red":
            img = img[0, ...]
        elif channel == "green":
            img = img[1, ...]
        elif channel == "blue":
            img = img[2, ...]
        else:
            img = img[:3, ...].max(axis=0)

    # If there are additional planes, collapse them by max projection.
    elif img.ndim > 2:
        img = img.reshape((-1, img.shape[-2], img.shape[-1])).max(axis=0)

    if img.ndim != 2:
        raise ValueError(f"Could not reduce TIFF to 2D. Final shape: {img.shape}")

    return img.astype(float)


def normalize_for_display(img):
    p_low, p_high = np.percentile(img, [LOWER_PERCENTILE, UPPER_PERCENTILE])
    if p_high <= p_low:
        p_high = p_low + 1.0
    out = (img - p_low) / (p_high - p_low)
    return np.clip(out, 0, 1)


# =========================
# SIMPLE MORPHOLOGY
# =========================

def erode_binary(mask_bool, iterations=1):
    out = mask_bool.copy()

    for _ in range(iterations):
        p = np.pad(out, 1, mode="constant", constant_values=False)

        neighbors = [
            p[0:-2, 0:-2],
            p[0:-2, 1:-1],
            p[0:-2, 2:],
            p[1:-1, 0:-2],
            p[1:-1, 1:-1],
            p[1:-1, 2:],
            p[2:, 0:-2],
            p[2:, 1:-1],
            p[2:, 2:],
        ]

        out = np.logical_and.reduce(neighbors)

    return out


def dilate_binary(mask_bool, iterations=1):
    out = mask_bool.copy()

    for _ in range(iterations):
        p = np.pad(out, 1, mode="constant", constant_values=False)

        out = (
            p[0:-2, 0:-2] |
            p[0:-2, 1:-1] |
            p[0:-2, 2:] |
            p[1:-1, 0:-2] |
            p[1:-1, 1:-1] |
            p[1:-1, 2:] |
            p[2:, 0:-2] |
            p[2:, 1:-1] |
            p[2:, 2:]
        )

    return out


def make_inner_ring(cell_mask, width_pixels=3):
    eroded = erode_binary(cell_mask, iterations=width_pixels)
    return cell_mask & (~eroded)


def mask_boundaries(masks):
    """
    Boundary of every nonzero Cellpose mask.
    The saved mask itself defines which cells are included.
    """
    b = np.zeros_like(masks, dtype=bool)

    b[:-1, :] |= masks[:-1, :] != masks[1:, :]
    b[:, :-1] |= masks[:, :-1] != masks[:, 1:]

    b[masks == 0] = False
    return b


def thicken_boundaries(boundaries, thickness=2):
    if thickness <= 1:
        return boundaries
    return dilate_binary(boundaries, iterations=thickness - 1)


# =========================
# POLARITY
# =========================

def angle_from_vector_anterior_top(vx, vy):
    """
    Image coordinates:
      +x = right
      +y = down

    Return:
      0 = up/anterior
      +90 = right
      +/-180 = down/posterior
      -90 = left
    """
    angle = np.degrees(np.arctan2(vx, -vy))

    if angle > 180:
        angle -= 360
    if angle <= -180:
        angle += 360

    return float(angle)


def quantify_cell_polarity(cell_id, masks, signal_img):
    cell_mask = masks == cell_id
    area = int(cell_mask.sum())

    if area < MIN_CELL_AREA_PIXELS:
        return None

    ys_cell, xs_cell = np.nonzero(cell_mask)
    cy = float(np.mean(ys_cell))
    cx = float(np.mean(xs_cell))

    ring = make_inner_ring(cell_mask, width_pixels=RING_WIDTH_PIXELS)
    ys, xs = np.nonzero(ring)

    ring_n = len(xs)

    if ring_n < MIN_RING_PIXELS:
        return None

    values = signal_img[ys, xs].astype(float)

    ring_mean = float(np.mean(values))
    ring_median = float(np.median(values))
    ring_std = float(np.std(values))

    # Preserve previous analysis:
    # only positive enrichment above each cell's ring mean contributes.
    weights = np.clip(values - ring_mean, 0, None)
    weight_sum = float(np.sum(weights))

    if weight_sum <= 1e-12:
        return None

    dx = xs.astype(float) - cx
    dy = ys.astype(float) - cy

    radius = np.sqrt(dx**2 + dy**2)
    valid = radius > 0

    dx = dx[valid]
    dy = dy[valid]
    weights = weights[valid]
    radius = radius[valid]

    ux = dx / radius
    uy = dy / radius

    vx = float(np.sum(weights * ux) / np.sum(weights))
    vy = float(np.sum(weights * uy) / np.sum(weights))

    polarity_strength = float(np.sqrt(vx**2 + vy**2))
    angle_deg = angle_from_vector_anterior_top(vx, vy)

    return {
        "cell_id": int(cell_id),
        "area_pixels": area,
        "centroid_x": cx,
        "centroid_y": cy,
        "ring_pixels": ring_n,
        "vangl2_ring_mean": ring_mean,
        "vangl2_ring_median": ring_median,
        "vangl2_ring_std": ring_std,
        "weight_sum": weight_sum,
        "vector_x": vx,
        "vector_y": vy,
        "polarity_strength": polarity_strength,
        "angle_deg_anterior_0": angle_deg,
    }


def rayleigh_from_angles_deg(angles_deg):
    """
    Standard Rayleigh test approximation for non-uniformity.
    Returns n, mean direction (deg, 0=anterior), R, z, p.
    """
    theta = np.deg2rad(np.asarray(angles_deg, dtype=float))
    n = len(theta)

    if n == 0:
        return 0, np.nan, np.nan, np.nan, np.nan

    C = np.sum(np.cos(theta))
    S = np.sum(np.sin(theta))

    Rsum = np.sqrt(C**2 + S**2)
    R = Rsum / n
    mean_angle = np.degrees(np.arctan2(S, C)) % 360
    z = n * R**2

    # Rayleigh p approximation.
    p = np.exp(
        np.sqrt(1 + 4*n + 4*(n**2 - Rsum**2))
        - (1 + 2*n)
    )

    if n < 50:
        if n < 10:
            p *= (
                1
                + (2*z - z**2) / (4*n)
                - (
                    24*z
                    - 132*z**2
                    + 76*z**3
                    - 9*z**4
                ) / (288*n**2)
            )
        else:
            p *= 1 + (2*z - z**2) / (4*n)

    p = float(np.clip(p, 0, 1))

    return n, float(mean_angle), float(R), float(z), p


def summarize(rows, total_rois):
    angles = [r["angle_deg_anterior_0"] for r in rows]
    n, mean_angle, R, z, rayleigh_p = rayleigh_from_angles_deg(angles)

    strengths = np.array([r["polarity_strength"] for r in rows], dtype=float)

    return {
        "total_cellpose_rois_in_saved_mask": int(total_rois),
        "n_cells_quantified": int(n),
        "n_cells_removed_by_basic_filters": int(total_rois - n),
        "mean_direction_deg_0_is_anterior": mean_angle,
        "mean_resultant_length_R": R,
        "rayleigh_z": z,
        "rayleigh_p_value": rayleigh_p,
        "mean_cell_polarity_strength": float(np.mean(strengths)),
        "median_cell_polarity_strength": float(np.median(strengths)),
        "automatic_edge_exclusion": "NO",
        "edge_cell_rule": "Saved Cellpose mask is definitive",
    }


# =========================
# OUTPUTS
# =========================

def save_cell_csv(rows, output_path):
    fieldnames = [
        "cell_id",
        "area_pixels",
        "centroid_x",
        "centroid_y",
        "ring_pixels",
        "vangl2_ring_mean",
        "vangl2_ring_median",
        "vangl2_ring_std",
        "weight_sum",
        "vector_x",
        "vector_y",
        "polarity_strength",
        "angle_deg_anterior_0",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_csv(summary, output_path):
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def make_yellow_overlay_rgb(signal_img, masks):
    base = normalize_for_display(signal_img)
    rgb = np.dstack([base, base, base])

    boundaries = thicken_boundaries(
        mask_boundaries(masks),
        thickness=BOUNDARY_THICKNESS
    )

    rgb[boundaries] = BOUNDARY_RGB
    return rgb


def save_mask_overlay(signal_img, masks, output_path):
    rgb = make_yellow_overlay_rgb(signal_img, masks)

    fig, ax = plt.subplots(figsize=(8, 12))
    ax.imshow(rgb)
    ax.set_title("Saved Cellpose masks used for Vangl2 polarity quantification")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_vector_overlay(signal_img, masks, rows, output_path):
    rgb = make_yellow_overlay_rgb(signal_img, masks)

    fig, ax = plt.subplots(figsize=(8, 12))
    ax.imshow(rgb)

    for r in rows:
        cx = r["centroid_x"]
        cy = r["centroid_y"]
        vx = r["vector_x"]
        vy = r["vector_y"]
        strength = r["polarity_strength"]

        if strength <= 1e-12:
            continue

        dx_arrow = ARROW_LENGTH_PIXELS * vx / strength
        dy_arrow = ARROW_LENGTH_PIXELS * vy / strength

        ax.arrow(
            cx,
            cy,
            dx_arrow,
            dy_arrow,
            color=ARROW_COLOR,
            width=0.6,
            head_width=4,
            head_length=4,
            length_includes_head=True,
            alpha=0.9,
        )

    ax.text(
        0.5, 0.985, "ANTERIOR",
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=12, fontweight="bold",
        color="white",
        bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=2)
    )

    ax.text(
        0.5, 0.015, "POSTERIOR",
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=12, fontweight="bold",
        color="white",
        bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=2)
    )

    ax.set_title("Vangl2 polarity vectors; anterior = top")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_rose_plot(rows, summary, output_path):
    angles_deg = np.array([r["angle_deg_anterior_0"] for r in rows], dtype=float)
    angles_rad = np.deg2rad(angles_deg)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="polar")

    bins = np.linspace(-np.pi, np.pi, 19)
    ax.hist(angles_rad, bins=bins)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ax.set_title(
        "Vangl2 polarity angles\n"
        f"n={summary['n_cells_quantified']}, "
        f"R={summary['mean_resultant_length_R']:.3f}, "
        f"Rayleigh p={summary['rayleigh_p_value']:.3g}"
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mask",
        help="Cellpose *_seg.npy file"
    )

    parser.add_argument(
        "--signal",
        help="Vangl2 TIFF file"
    )

    parser.add_argument(
        "--name",
        help="Output prefix. Default: TIFF stem"
    )

    parser.add_argument(
        "--channel",
        choices=["red", "green", "blue", "max"],
        default="red",
        help="Signal channel to quantify; default = red"
    )

    args = parser.parse_args()

    folder = Path(__file__).resolve().parent

    # If explicit filenames are not given, auto-detect only if unambiguous.
    if args.mask:
        mask_path = Path(args.mask)
        if not mask_path.is_absolute():
            mask_path = folder / mask_path
    else:
        masks = sorted(folder.glob("*_seg.npy"))
        if len(masks) != 1:
            raise RuntimeError(
                f"Found {len(masks)} *_seg.npy files. "
                "Please specify one with --mask."
            )
        mask_path = masks[0]

    if args.signal:
        signal_path = Path(args.signal)
        if not signal_path.is_absolute():
            signal_path = folder / signal_path
    else:
        tifs = sorted(list(folder.glob("*.tif")) + list(folder.glob("*.tiff")))
        if len(tifs) != 1:
            raise RuntimeError(
                f"Found {len(tifs)} TIFF files. "
                "Please specify one with --signal."
            )
        signal_path = tifs[0]

    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    if not signal_path.exists():
        raise FileNotFoundError(signal_path)

    name = args.name if args.name else signal_path.stem

    signal_img = load_image_2d(signal_path, channel=args.channel)
    masks = load_cellpose_masks(mask_path)

    print("Signal image shape:", signal_img.shape)
    print("Mask shape:", masks.shape)

    if signal_img.shape != masks.shape:
        raise ValueError(
            f"Image and mask shapes do not match: "
            f"{signal_img.shape} vs {masks.shape}"
        )

    labels = [int(x) for x in np.unique(masks) if int(x) != 0]
    total_rois = len(labels)

    print("ROIs in saved Cellpose mask:", total_rois)
    print("Automatic edge exclusion: NO")
    print("Cells retained in the saved mask are kept.")

    rows = []

    for cell_id in labels:
        result = quantify_cell_polarity(cell_id, masks, signal_img)
        if result is not None:
            rows.append(result)

    if not rows:
        raise RuntimeError("No cells passed the basic area/ring filters.")

    summary = summarize(rows, total_rois)

    cell_csv = folder / f"{name}_vangl2_polarity_cells.csv"
    summary_csv = folder / f"{name}_vangl2_polarity_summary.csv"
    mask_png = folder / f"{name}_mask_overlay_YELLOW.png"
    vector_png = folder / f"{name}_vangl2_polarity_vectors.png"
    rose_png = folder / f"{name}_vangl2_polarity_roseplot.png"

    save_cell_csv(rows, cell_csv)
    save_summary_csv(summary, summary_csv)
    save_mask_overlay(signal_img, masks, mask_png)
    save_vector_overlay(signal_img, masks, rows, vector_png)
    save_rose_plot(rows, summary, rose_png)

    print("\nSummary")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nSaved:")
    print(cell_csv.name)
    print(summary_csv.name)
    print(mask_png.name)
    print(vector_png.name)
    print(rose_png.name)


if __name__ == "__main__":
    main()
