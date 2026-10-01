import os
import re
import csv
import math
import numpy as np
import cv2
from PIL import Image

Image.MAX_IMAGE_PIXELS = None




# ============================================================
# Step 0) threshold DAB tree -> binary masks + thresholded RGBA
# ============================================================
def dab_to_binary_mask(
    dab_png_path: str,
    output_folder: str,
    low: int = 0,
    high: int = 220,
) -> str:
    """
    Convert one DAB single-channel PNG to a binary threshold mask.

    Rules:
      - grayscale intensity in [low, high] -> 255 (positive)
      - otherwise -> 0
      - when the input is RGBA, pixels outside the original ROI (alpha==0)
        are forced to 0 in the exported mask.

    Output filename:
      xxx.png -> xxx_mask.png
    """
    if not dab_png_path.lower().endswith(".png"):
        raise ValueError("Input must be a PNG file")
    if not (0 <= int(low) <= 255 and 0 <= int(high) <= 255):
        raise ValueError("low/high must be within 0..255")
    if int(low) > int(high):
        raise ValueError("low must be <= high")

    os.makedirs(output_folder, exist_ok=True)

    src = Image.open(dab_png_path)
    gray = np.asarray(src.convert("L"), dtype=np.uint8)
    positive = (gray >= int(low)) & (gray <= int(high))

    # Respect the existing ROI alpha when present. This keeps the exported
    # binary mask itself limited to the valid ROI, while preserving the
    # same threshold rule used by the original notebook.
    if "A" in src.getbands():
        alpha = np.asarray(src.convert("RGBA"), dtype=np.uint8)[..., 3]
        positive &= alpha > 0

    mask = positive.astype(np.uint8) * 255

    base = os.path.splitext(os.path.basename(dab_png_path))[0]
    out_path = os.path.join(output_folder, base + "_mask.png")
    Image.fromarray(mask, mode="L").save(out_path)
    return out_path


def apply_mask_to_rgba_dab(
    rgba_png_path: str,
    mask_png_path: str,
    output_folder: str,
    threshold_value: int = 220,
) -> str:
    """
    Apply a binary mask to the DAB RGBA image while retaining the original
    grayscale/RGB intensity at positive pixels. Negative pixels become fully
    transparent (0,0,0,0).

    Output filename:
      xxx_mask.png -> xxx_threshold220.png
    """
    os.makedirs(output_folder, exist_ok=True)

    rgba = np.asarray(Image.open(rgba_png_path).convert("RGBA"), dtype=np.uint8)
    mask = np.asarray(Image.open(mask_png_path).convert("L"), dtype=np.uint8)

    if rgba.shape[:2] != mask.shape[:2]:
        raise RuntimeError(
            f"Size mismatch between DAB and mask: {rgba_png_path} {rgba.shape[:2]} vs "
            f"{mask_png_path} {mask.shape[:2]}"
        )

    keep = mask == 255
    out = np.zeros_like(rgba, dtype=np.uint8)
    out[keep] = rgba[keep]

    base = os.path.splitext(os.path.basename(rgba_png_path))[0]
    out_name = f"{base}_threshold{int(threshold_value):03d}.png"
    out_path = os.path.join(output_folder, out_name)
    Image.fromarray(out, mode="RGBA").save(out_path)
    return out_path


def batch_threshold_dab_folder_tree(
    dab_root: str,
    mask_root: str,
    threshold_root: str,
    low: int = 0,
    high: int = 220,
) -> tuple[int, int]:
    """
    Recursively process every PNG under ``dab_root``.

    For every input DAB PNG, this function creates two synchronized outputs:
      1) mask_root/<relative tree>/xxx_mask.png
      2) threshold_root/<relative tree>/xxx_threshold###.png

    The thresholded RGBA keeps the original DAB intensity only at positive
    pixels and is therefore directly suitable for the existing Part 3 single-
    marker quantification and Part 4 colocalization code.
    """
    if not os.path.isdir(dab_root):
        raise ValueError(f"DAB root does not exist: {dab_root}")

    os.makedirs(mask_root, exist_ok=True)
    os.makedirs(threshold_root, exist_ok=True)

    processed = 0
    failed = 0

    for root, _, files in os.walk(dab_root):
        rel_dir = os.path.relpath(root, dab_root)
        mask_out_dir = mask_root if rel_dir == "." else os.path.join(mask_root, rel_dir)
        thr_out_dir = threshold_root if rel_dir == "." else os.path.join(threshold_root, rel_dir)
        os.makedirs(mask_out_dir, exist_ok=True)
        os.makedirs(thr_out_dir, exist_ok=True)

        for fn in sorted(files):
            if not fn.lower().endswith(".png"):
                continue

            dab_path = os.path.join(root, fn)
            try:
                mask_path = dab_to_binary_mask(
                    dab_png_path=dab_path,
                    output_folder=mask_out_dir,
                    low=low,
                    high=high,
                )
                threshold_path = apply_mask_to_rgba_dab(
                    rgba_png_path=dab_path,
                    mask_png_path=mask_path,
                    output_folder=thr_out_dir,
                    threshold_value=high,
                )
                processed += 1
                print(f"[THRESHOLD OK] {threshold_path}")
            except Exception as e:
                failed += 1
                print(f"[THRESHOLD FAIL] {dab_path} :: {e}")

    print(f"\nTHRESHOLDING DONE. processed={processed}, failed={failed}")
    return processed, failed


# ============================================================
# Step 1) find threshold PNGs (degenerate symbol ??? supported)
# ============================================================
def list_threshold_pngs(
    threshold_root: str,
    holder1_regex: str = r"threshold\d{3}\.png",
) -> list[str]:
    """
    Return full paths of all PNGs under threshold_root (recursive)
    whose filename matches holder1_regex (regex on basename).
    Default: threshold###.png (### are 3 digits)
    """
    if not os.path.isdir(threshold_root):
        raise ValueError(f"Threshold root does not exist: {threshold_root}")

    pat = re.compile(holder1_regex, flags=re.IGNORECASE)
    out = []
    for root, _, files in os.walk(threshold_root):
        for fn in files:
            if not fn.lower().endswith(".png"):
                continue
            if pat.search(fn):
                out.append(os.path.join(root, fn))
    return out


# ============================================================
# Step 2) pair threshold pngs with DAB pngs (same tree)
# ============================================================
def build_threshold_dab_pairs(
    dab_root: str,
    threshold_root: str,
    holder1_regex: str = r"threshold\d{3}\.png",
) -> list[tuple[str, str]]:
    """
    For each threshold png in threshold_root matching holder1_regex,
    find corresponding DAB png in dab_root with same relative folder.

    Filename mapping:
      - Find the matched holder string in basename (e.g. 'threshold220.png')
      - Remove the preceding '_' or '-' + holder_str:
          xxx_threshold220.png -> xxx.png
          xxx-DAB_threshold220.png -> xxx-DAB.png
    """
    if not os.path.isdir(dab_root):
        raise ValueError(f"DAB root does not exist: {dab_root}")
    if not os.path.isdir(threshold_root):
        raise ValueError(f"Threshold root does not exist: {threshold_root}")

    thr_paths = list_threshold_pngs(threshold_root, holder1_regex=holder1_regex)
    pat = re.compile(holder1_regex, flags=re.IGNORECASE)

    pairs: list[tuple[str, str]] = []
    missing = 0
    skipped = 0

    for thr_path in thr_paths:
        rel = os.path.relpath(thr_path, threshold_root)
        rel_dir = os.path.dirname(rel)
        thr_base = os.path.basename(thr_path)

        m = pat.search(thr_base)
        if not m:
            skipped += 1
            continue

        holder_str = m.group(0)  # e.g. threshold220.png

        # Remove suffix like "_threshold220.png" or "-threshold220.png" -> ".png"
        dab_base = re.sub(
            rf"([_-]){re.escape(holder_str)}$",
            ".png",
            thr_base,
            flags=re.IGNORECASE,
        )

        # Fallback if no separator matched: replace holder_str with ".png"
        if dab_base == thr_base:
            dab_base = re.sub(
                rf"{re.escape(holder_str)}$",
                ".png",
                thr_base,
                flags=re.IGNORECASE,
            )

        dab_path = os.path.join(dab_root, rel_dir, dab_base)
        if not os.path.exists(dab_path):
            missing += 1
            print(f"[MISSING DAB] {dab_path}")
            continue

        pairs.append((thr_path, dab_path))

    print(
        f"\nPAIRING DONE. pairs={len(pairs)}, missing_dab={missing}, "
        f"skipped={skipped}, total_threshold_found={len(thr_paths)}"
    )
    return pairs


# ============================================================
# helpers
# ============================================================
def _load_rgba(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGBA"))

def _rgb_to_gray_float(rgba: np.ndarray) -> np.ndarray:
    """
    Convert RGBA->grayscale luminance from RGB.
    Output float64 [0..255].
    """
    rgb = rgba[:, :, :3].astype(np.float64)
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

def _extract_threshold_value_from_name(basename: str) -> int | None:
    """
    Find 3 digits right after 'threshold' in filename, e.g. threshold220 -> 220.
    """
    m = re.search(r"threshold(\d{3})", basename, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


# ============================================================
# Step 3) quantify threshold RGBA (positive = non-transparent)
# ============================================================
def quantify_threshold_rgba(
    threshold_rgba_path: str
) -> list[tuple[str, object]]:
    """
    Threshold RGBA: positive region is non-transparent (alpha > 0).

    Intensity stats are computed over positive pixels only (grayscale 0..255).
    OD metrics computed over positive pixels only:

      OD_from_meanI = log10(255 / mean(I_pos))
      AOD           = mean( log10(255 / I_pos) )
      IOD           = sum(  log10(255 / I_pos) )

    Note: AOD != OD_from_meanI due to log nonlinearity.
    """
    rgba = _load_rgba(threshold_rgba_path)
    alpha = rgba[:, :, 3]
    pos_mask = alpha > 0

    base = os.path.basename(threshold_rgba_path)
    thr_val = _extract_threshold_value_from_name(base)

    pos_area = int(pos_mask.sum())

    gray = _rgb_to_gray_float(rgba)
    pos_vals = gray[pos_mask] if pos_area > 0 else np.array([], dtype=np.float64)

    if pos_area > 0:
        v_min = float(pos_vals.min())
        v_max = float(pos_vals.max())
        v_mean = float(pos_vals.mean())
        v_sum = float(pos_vals.sum())
    else:
        v_min = None
        v_max = None
        v_mean = None
        v_sum = 0.0

    # --- islands stats ---
    bin_img = (pos_mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    diameters = []
    circularities = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= 0:
            continue
        perim = cv2.arcLength(cnt, True)
        eq_d = 2.0 * math.sqrt(area / math.pi)
        diameters.append(eq_d)
        if perim > 0:
            circ = 4.0 * math.pi * area / (perim * perim)
            circularities.append(circ)

    island_count = int(len(diameters))
    avg_diameter = float(np.mean(diameters)) if diameters else 0.0
    avg_circularity = float(np.mean(circularities)) if circularities else 0.0

    # --- OD / AOD / IOD (positive pixels only) ---
    if pos_area > 0:
        I = pos_vals.astype(np.float64)
        I_safe = np.clip(I, 1e-6, 255.0)

        OD_pixels = np.log10(255.0 / I_safe)

        OD_from_meanI = float(np.log10(255.0 / float(np.mean(I_safe))))
        AOD = float(np.mean(OD_pixels))
        IOD = float(np.sum(OD_pixels))
    else:
        OD_from_meanI = None
        AOD = None
        IOD = 0.0

    rows = [
        ("threshold_filename", base),
        ("threshold_path", threshold_rgba_path),
        ("threshold_value", thr_val),

        ("positive_area_px", pos_area),
        ("intensity_min", v_min),
        ("intensity_mean", v_mean),
        ("intensity_max", v_max),
        ("integrated_intensity", v_sum),

        ("island_count", island_count),
        ("avg_island_equiv_diameter_px", avg_diameter),
        ("avg_island_circularity", avg_circularity),

        ("OD_from_meanI_log10_255_over_meanI", OD_from_meanI),
        ("AOD_mean_OD_pixels", AOD),
        ("IOD_sum_OD_pixels", IOD),
    ]
    return rows


# ============================================================
# Step 4) quantify DAB area (non-transparent)
# ============================================================
def quantify_dab_nontransparent_area(
    dab_path: str
) -> list[tuple[str, object]]:
    """
    ROI reference area:
      - If RGBA: alpha>0 area
      - Else: full image area
    """
    img = Image.open(dab_path)
    base = os.path.basename(dab_path)

    if img.mode != "RGBA":
        arr = np.array(img)
        h, w = arr.shape[:2]
        non_trans_area = int(h * w)
    else:
        rgba = np.array(img.convert("RGBA"))
        alpha = rgba[:, :, 3]
        non_trans_area = int((alpha > 0).sum())

    return [
        ("dab_filename", base),
        ("dab_path", dab_path),
        ("dab_nontransparent_area_px", non_trans_area),
    ]


# ============================================================
# CSV helpers
# ============================================================
def write_csv_header_if_needed(csv_path: str, header: list[str]) -> None:
    need_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
    if need_header:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

def append_csv_row(csv_path: str, row: list[object]) -> None:
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


# ============================================================
# Step 5) wrapper: quantify pairs -> add %POS(0-100) + products -> CSV
# ============================================================
def quantify_threshold_vs_dab_to_csv(
    dab_root: str,
    threshold_root: str,
    output_folder: str,
    holder1_regex: str = r"threshold\d{3}\.png",
    out_csv_name: str = "threshold_quantification.csv",
) -> str:
    """
    Full pipeline:
      - build pairs (threshold_path, dab_path)
      - quantify threshold + dab
      - POS_percent = positive_area / dab_area * 100
      - ODmean_x_POSpercent = OD_from_meanI * POS_percent
      - AOD_x_POSpercent    = AOD * POS_percent
      - write CSV
    """
    os.makedirs(output_folder, exist_ok=True)
    csv_path = os.path.join(output_folder, out_csv_name)

    pairs = build_threshold_dab_pairs(
        dab_root=dab_root,
        threshold_root=threshold_root,
        holder1_regex=holder1_regex,
    )

    if not pairs:
        print("No valid pairs found. CSV not written.")
        return csv_path

    # header based on first pair
    thr_rows0 = quantify_threshold_rgba(pairs[0][0])
    dab_rows0 = quantify_dab_nontransparent_area(pairs[0][1])

    thr_keys = [k for k, _ in thr_rows0]
    dab_keys = [k for k, _ in dab_rows0]

    extra_keys = [
        "POS_percent",                # 0..100
        "ODmean_x_POSpercent",        # OD_from_meanI * POS_percent
        "AOD_x_POSpercent",           # AOD * POS_percent
    ]

    header = thr_keys + dab_keys + extra_keys
    write_csv_header_if_needed(csv_path, header)

    for thr_path, dab_path in pairs:
        thr_rows = quantify_threshold_rgba(thr_path)
        dab_rows = quantify_dab_nontransparent_area(dab_path)

        thr = {k: v for k, v in thr_rows}
        dab = {k: v for k, v in dab_rows}

        pos_area = float(thr.get("positive_area_px", 0) or 0)
        dab_area = float(dab.get("dab_nontransparent_area_px", 0) or 0)

        POS_percent = (pos_area / dab_area * 100.0) if dab_area > 0 else None

        OD_from_meanI = thr.get("OD_from_meanI_log10_255_over_meanI", None)
        AOD = thr.get("AOD_mean_OD_pixels", None)

        ODmean_x_POS = (
            float(OD_from_meanI) * float(POS_percent)
            if (OD_from_meanI is not None and POS_percent is not None)
            else None
        )
        AOD_x_POS = (
            float(AOD) * float(POS_percent)
            if (AOD is not None and POS_percent is not None)
            else None
        )

        row = (
            [thr.get(k) for k in thr_keys]
            + [dab.get(k) for k in dab_keys]
            + [POS_percent, ODmean_x_POS, AOD_x_POS]
        )
        append_csv_row(csv_path, row)

    print(f"\nDONE. CSV saved: {csv_path}")
    return csv_path

import os
import re
import numpy as np
import pandas as pd


def csv_summary_by_marker_mouse_condition_layout_like_excel(
    csv_path: str,
    column_holder: str,
    marker_id_location: int,
    mouse_id_location: int,
    marker_holder: str,
    summary_target: str,
    csv_output_path: str,
    mouse_id_min: int = 1,
    mouse_id_max: int = 100,
    agg: str = "mean",
):
    """
    OUTPUT LAYOUT (matches your screenshot):

        A1 = marker_holder
        B1..E1 = IL-10+VEGF, IL-10, no cytokine, Adaptic  (condition columns)
        A2.. = mouse IDs (numeric order)
        Cells = aggregated summary_target for (mouse_id, condition)

    IMPORTANT FIX:
      Filenames often contain mouse-condition token like '4-1-roiRaw' not '4-1'.
      We parse leading pattern: r'^(\\d+)-(\\d)' and ignore suffix.
    """

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    for col in (column_holder, summary_target):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in CSV. Available: {list(df.columns)}")

    # ---------- helpers ----------
    def _basename_any(p):
        if pd.isna(p):
            return ""
        p = str(p)
        return os.path.basename(p.replace("/", os.sep).replace("\\", os.sep))

    def _underscore_token(fname: str, slot_1based: int) -> str:
        parts = fname.split("_")
        i = slot_1based
        if len(parts) <= i:
            return ""
        return parts[i]

    def _parse_mouse_condition(token: str):
        """
        Accept:
          '4-1' , '12-3', '4-1-roiRaw', '4-1-roiRaw_DAB.png', etc.
        Only parse the LEADING 'mouse-condition' and ignore suffix.
        """
        m = re.match(r"^\s*(\d{1,3})\s*-\s*(\d)\b", str(token))
        if not m:
            return None, None
        return int(m.group(1)), int(m.group(2))

    # ---------- parse ----------
    df["_fname"] = df[column_holder].apply(_basename_any)
    df["_marker"] = df["_fname"].apply(lambda s: _underscore_token(s, marker_id_location)).astype(str).str.strip()
    df = df[df["_marker"] == marker_holder].copy()

    mouse_token = df["_fname"].apply(lambda s: _underscore_token(s, mouse_id_location))
    parsed = mouse_token.apply(_parse_mouse_condition)
    df["_mouse_id"] = parsed.apply(lambda x: x[0])
    df["_cond_id"] = parsed.apply(lambda x: x[1])

    df = df.dropna(subset=["_mouse_id", "_cond_id"])
    df["_mouse_id"] = df["_mouse_id"].astype(int)
    df["_cond_id"] = df["_cond_id"].astype(int)

    df["_value"] = pd.to_numeric(df[summary_target], errors="coerce")

    # condition mapping + desired column order
    cond_name = {
        1: "IL-10+VEGF",
        3: "IL-10",
        2: "no cytokine",
        4: "Adaptic",
    }
    cond_order_ids = [1, 3, 2, 4]
    cond_cols = [cond_name[i] for i in cond_order_ids]

    # aggregate duplicates per (mouse, cond)
    if agg == "mean":
        g = df.groupby(["_mouse_id", "_cond_id"], dropna=False)["_value"].mean()
    elif agg == "median":
        g = df.groupby(["_mouse_id", "_cond_id"], dropna=False)["_value"].median()
    elif agg == "sum":
        g = df.groupby(["_mouse_id", "_cond_id"], dropna=False)["_value"].sum()
    elif agg == "first":
        g = df.groupby(["_mouse_id", "_cond_id"], dropna=False)["_value"].first()
    else:
        raise ValueError("agg must be one of: mean, median, sum, first")

    # build final table: rows=mice, cols=conditions
    mouse_ids = list(range(int(mouse_id_min), int(mouse_id_max) + 1))
    out = pd.DataFrame(index=mouse_ids, columns=cond_cols, dtype=float)

    for (mid, cid), val in g.items():
        if mid in out.index and cid in cond_name:
            out.loc[mid, cond_name[cid]] = float(val) if pd.notna(val) else np.nan

    # write EXACT layout you showed
    rows = []
    rows.append([marker_holder] + cond_cols)  # header row (A1 is marker)
    for mid in mouse_ids:
        vals = []
        for c in cond_cols:
            v = out.loc[mid, c]
            vals.append("" if pd.isna(v) else v)
        rows.append([mid] + vals)

    os.makedirs(os.path.dirname(os.path.abspath(csv_output_path)), exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_output_path, header=False, index=False)
    return csv_output_path


def summarize_all_markers_layout_like_excel(
    input_csv_path: str,
    summary_target: str,
    output_folder_path: str,
    column_holder: str = "dab_path",
    marker_id_location: int = 1,
    mouse_id_location: int = 2,
    mouse_id_min: int = 1,
    mouse_id_max: int = 100,
    agg: str = "mean",
):
    """
    Wrapper:
    - detects all marker holders present in the filename token at marker_id_location
    - produces one output CSV per marker with the Excel-like layout
    """
    if not os.path.isfile(input_csv_path):
        raise FileNotFoundError(f"Input CSV not found: {input_csv_path}")
    os.makedirs(output_folder_path, exist_ok=True)

    df = pd.read_csv(input_csv_path)
    if column_holder not in df.columns:
        raise ValueError(f"Missing column '{column_holder}' in CSV. Available: {list(df.columns)}")
    if summary_target not in df.columns:
        raise ValueError(f"Missing column '{summary_target}' in CSV. Available: {list(df.columns)}")

    def _basename_any(p):
        if pd.isna(p):
            return ""
        p = str(p)
        return os.path.basename(p.replace("/", os.sep).replace("\\", os.sep))

    def _underscore_token(fname: str, slot_1based: int) -> str:
        parts = fname.split("_")
        i = slot_1based
        if len(parts) <= i:
            return ""
        return parts[i]

    fnames = df[column_holder].apply(_basename_any)
    markers = (
        fnames.apply(lambda s: _underscore_token(s, marker_id_location))
        .astype(str)
        .str.strip()
    )
    markers = sorted({m for m in markers.tolist() if m and m.lower() != "nan"})

    written = []
    for marker in markers:
        safe_marker = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in marker)
        safe_target = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in summary_target)
        out_path = os.path.join(output_folder_path, f"{safe_marker}__{safe_target}__summary.csv")

        csv_summary_by_marker_mouse_condition_layout_like_excel(
            csv_path=input_csv_path,
            column_holder=column_holder,
            marker_id_location=marker_id_location,
            mouse_id_location=mouse_id_location,
            marker_holder=marker,
            summary_target=summary_target,
            csv_output_path=out_path,
            mouse_id_min=mouse_id_min,
            mouse_id_max=mouse_id_max,
            agg=agg,
        )
        written.append(out_path)

    return written

def _select_directory(title: str) -> str:
    """Open a folder-selection dialog and return the selected absolute path."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title=title, mustexist=True)
    root.destroy()

    if not folder:
        raise SystemExit("[INFO] Folder selection cancelled by user.")
    return os.path.abspath(folder)


if __name__ == "__main__":
    # Part 3 now starts from the DAB single-channel folder tree.
    # It generates masks + thresholded RGBA images first, then performs
    # the existing single-channel quantification on the generated images.
    dab_root = _select_directory(
        "Part 3 - Select INPUT DAB single-channel root folder"
    )
    output_folder = _select_directory(
        "Part 3 - Select OUTPUT root folder"
    )

    # Threshold settings retained from the original notebooks.
    threshold_low = 0
    threshold_high = 220

    mask_root = os.path.join(output_folder, "Mask")
    threshold_root = os.path.join(output_folder, f"Thresholded_{threshold_high:03d}")

    # 1) Generate binary masks and masked/thresholded RGBA images.
    processed, failed = batch_threshold_dab_folder_tree(
        dab_root=dab_root,
        mask_root=mask_root,
        threshold_root=threshold_root,
        low=threshold_low,
        high=threshold_high,
    )

    # 2) Quantify the newly generated thresholded images against the
    #    corresponding original DAB images.
    quant_csv = quantify_threshold_vs_dab_to_csv(
        dab_root=dab_root,
        threshold_root=threshold_root,
        output_folder=output_folder,
        holder1_regex=rf"threshold{threshold_high:03d}\.png",
        out_csv_name=f"DABsingle_threshold{threshold_high:03d}_quantification.csv",
    )

    # 3) Keep the original summary stage, now driven directly by the CSV
    #    generated above rather than by any fixed input path.
    summary_output_folder = os.path.join(
        output_folder, f"summary_AOD_x_POSpercent_threshold{threshold_high:03d}"
    )
    outs = summarize_all_markers_layout_like_excel(
        input_csv_path=quant_csv,
        summary_target="AOD_x_POSpercent",
        output_folder_path=summary_output_folder,
        column_holder="dab_path",
        marker_id_location=1,
        mouse_id_location=2,
        mouse_id_min=4,
        mouse_id_max=13,
    )

    print("\n" + "=" * 70)
    print("[INFO] Part 3 finished.")
    print(f"[INFO] Thresholded images created : {processed}")
    print(f"[INFO] Threshold failures         : {failed}")
    print(f"[INFO] Mask folder                : {mask_root}")
    print(f"[INFO] Thresholded image folder   : {threshold_root}")
    print(f"[INFO] Quantification CSV         : {quant_csv}")
    print(f"[INFO] Summary folder             : {summary_output_folder}")
    print("=" * 70)
