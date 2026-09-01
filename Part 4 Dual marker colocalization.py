"""
Part IV: Dual-Marker Immunohistochemistry Colocalization Analysis Pipeline
This part quantifies spatial colocalization using various scores between two immunohistochemical markers using registered-and-aligned threshold-masked RGBA positive-region images and a DAB ROI-corresponding image (can be registered-and-aligned image of one of the 2 compared marker). Positive regions are defined by non-transparent alpha pixels. It provides image loading, alpha-based mask generation, OD map reconstruction, and calculation of global and local colocalization metrics including overlap, directional ratios, Jaccard, Dice, Phi, Pearson correlation, and weighted Jaccard. Tile-based analysis and correlation of positive area fractions are supported. The script handles size mismatches by common-minimum cropping, supports folder-based batch processing of marker triplets, and includes a single-triplet demo. Paired files are matched by the shared core key <marker> amone filename: <slide_index>_<marker>_<sample name>_threshold<thresholded value>.png. Dependencies: Python ≥3.7, numpy, opencv-python, Pillow. Author: Zonghan Gan"""
import os
import re
import csv
import math
import numpy as np
import cv2
Image.MAX_IMAGE_PIXELS = None
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
# =============================================================================
# 1) BASIC FUNCTIONS (DEFINED ONCE)
# =============================================================================
def load_rgba(path: str) -> np.ndarray:
    """Load RGBA PNG as uint8 array [H,W,4]."""
    return np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)
def alpha_mask(rgba: np.ndarray, alpha_threshold: int = 1) -> np.ndarray:
    """Return boolean mask where alpha >= alpha_threshold."""
    return rgba[..., 3] >= alpha_threshold
def common_crop3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop three images to their common min H,W."""
    h = min(a.shape[0], b.shape[0], c.shape[0])
    w = min(a.shape[1], b.shape[1], c.shape[1])
    return a[:h, :w], b[:h, :w], c[:h, :w]
def rgb_to_gray(rgb_uint8: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 image to grayscale float32 [0..255] using luminance."""
    r = rgb_uint8[..., 0].astype(np.float32)
    g = rgb_uint8[..., 1].astype(np.float32)
    b = rgb_uint8[..., 2].astype(np.float32)
    return 0.299 * r + 0.587 * g + 0.114 * b
def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation; returns NaN if not enough points or zero variance."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 2:
        return float("nan")
    sx = np.nanstd(x)
    sy = np.nanstd(y)
    if not np.isfinite(sx) or not np.isfinite(sy) or sx == 0.0 or sy == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])
import math
def phi_coefficient(x_bool: np.ndarray, y_bool: np.ndarray) -> float:
    """
    Phi coefficient = Pearson correlation for binary variables.
    Input arrays are 1D boolean vectors (already ROI-filtered).
    """
    x = np.asarray(x_bool, dtype=bool)
    y = np.asarray(y_bool, dtype=bool)
    tp = int(np.logical_and(x, y).sum())
    tn = int(np.logical_and(~x, ~y).sum())
    fp = int(np.logical_and(~x, y).sum())
    fn = int(np.logical_and(x, ~y).sum())
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom <= 0:
        return float("nan")
    return (tp * tn - fp * fn) / math.sqrt(float(denom))
def nan_stats(v: List[float]) -> Tuple[float, float, float]:
    """Return (mean, median, std) with NaN handling."""
    arr = np.asarray(v, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.nanmean(arr)), float(np.nanmedian(arr)), float(np.nanstd(arr))
def compute_od_map_from_thresholded_rgba(
    marker_rgba: np.ndarray,
    marker_pos: np.ndarray,
    roi_mask: np.ndarray,
    *,
    od_eps: float = 1.0,
    i0_percentile: float = 95.0,
    i0_fallback: float = 240.0,
) -> Tuple[np.ndarray, float]:
    """
    OD map for chromogenic IHC given thresholded RGBA:
      - marker_pos pixels keep RGB intensity (stain strength)
      - non-marker pixels are transparent (no intensity), so we assign them I0 -> OD=0
      - I0 is estimated from ROI pixels where marker_pos is False (ROI background).
    Returns (od_map_float32, i0_used).
    """
    gray = rgb_to_gray(marker_rgba[..., :3])  # float32 0..255
    bg = np.logical_and(roi_mask, ~marker_pos)
    if int(bg.sum()) > 0:
        i0 = float(np.percentile(gray[bg], i0_percentile))
    else:
        i0 = float(i0_fallback)
    I = np.where(marker_pos, gray, i0).astype(np.float32)
    od = -np.log10((I + od_eps) / (i0 + od_eps)).astype(np.float32)
    od[od < 0] = 0.0
    od = np.where(roi_mask, od, 0.0).astype(np.float32)
    return od, i0
def quantify_triplet_core(
    marker1: str,
    marker2: str,
    marker1_png_path: str,
    marker2_png_path: str,
    dab_png_path: str,
    *,
    tile_size: int = 256,
    alpha_threshold: int = 1,
    allow_size_mismatch: bool = False,
    # OD options
    od_eps: float = 1.0,
    i0_percentile: float = 95.0,
    i0_fallback: float = 240.0,
) -> Dict[str, object]:
    """
    Compute all requested metrics for ONE aligned triplet:
      - Binary overlap: directional ratios, Jaccard, Dice
      - global_phi
      - Global OD Pearson correlation
      - Weighted Jaccard (continuous): sum min(OD1,OD2)/sum max(OD1,OD2)
      - Tile-based OD correlation (mean/median/std)
      - tile_area_correlation: corr across tiles of marker area fraction
      - plus tile jaccard summaries
    """
    m1_rgba = load_rgba(marker1_png_path)
    m2_rgba = load_rgba(marker2_png_path)
    roi_rgba = load_rgba(dab_png_path)
    if (m1_rgba.shape[:2] != m2_rgba.shape[:2]) or (m1_rgba.shape[:2] != roi_rgba.shape[:2]):
        if not allow_size_mismatch:
            raise RuntimeError(
                "Size mismatch:\n"
                f"  m1 : {marker1_png_path} {m1_rgba.shape}\n"
                f"  m2 : {marker2_png_path} {m2_rgba.shape}\n"
                f"  roi: {dab_png_path} {roi_rgba.shape}\n"
            )
        m1_rgba, m2_rgba, roi_rgba = common_crop3(m1_rgba, m2_rgba, roi_rgba)
    roi = alpha_mask(roi_rgba, alpha_threshold)
    m1_pos = alpha_mask(m1_rgba, alpha_threshold)
    m2_pos = alpha_mask(m2_rgba, alpha_threshold)
    # Binary inside ROI
    m1r = np.logical_and(m1_pos, roi)
    m2r = np.logical_and(m2_pos, roi)
    roi_area = int(roi.sum())
    m1_area = int(m1r.sum())
    m2_area = int(m2r.sum())
    overlap = int(np.logical_and(m1r, m2r).sum())
    union = int(np.logical_or(m1r, m2r).sum())
    m1_over_m2 = (overlap / m2_area) if m2_area > 0 else float("nan")
    m2_over_m1 = (overlap / m1_area) if m1_area > 0 else float("nan")
    jaccard = (overlap / union) if union > 0 else float("nan")
    dice = (2.0 * overlap / (m1_area + m2_area)) if (m1_area + m2_area) > 0 else float("nan")
    # global_phi on ROI pixels
    roi_idx = roi.reshape(-1)
    global_phi = phi_coefficient(m1_pos.reshape(-1)[roi_idx], m2_pos.reshape(-1)[roi_idx])
    # OD maps + global OD metrics
    od1, i0_1 = compute_od_map_from_thresholded_rgba(
        m1_rgba, m1_pos, roi, od_eps=od_eps, i0_percentile=i0_percentile, i0_fallback=i0_fallback
    )
    od2, i0_2 = compute_od_map_from_thresholded_rgba(
        m2_rgba, m2_pos, roi, od_eps=od_eps, i0_percentile=i0_percentile, i0_fallback=i0_fallback
    )
    od1_roi = od1.reshape(-1)[roi_idx]
    od2_roi = od2.reshape(-1)[roi_idx]
    global_od_pearson = pearson_corr(od1_roi, od2_roi)
    min_sum = float(np.minimum(od1_roi, od2_roi).sum())
    max_sum = float(np.maximum(od1_roi, od2_roi).sum())
    weighted_jaccard_continuous = (min_sum / max_sum) if max_sum > 0 else float("nan")
    # Tiles
    H, W = roi.shape
    ts = int(tile_size)
    tile_jacc: List[float] = []
    tile_od_corr: List[float] = []
    tile_m1_area_frac: List[float] = []
    tile_m2_area_frac: List[float] = []
    tiles_used = 0
    for y0 in range(0, H, ts):
        y1 = min(H, y0 + ts)
        for x0 in range(0, W, ts):
            x1 = min(W, x0 + ts)
            roi_t = roi[y0:y1, x0:x1]
            roi_t_area = int(roi_t.sum())
            if roi_t_area == 0:
                continue
            tiles_used += 1
            m1_t = m1r[y0:y1, x0:x1]
            m2_t = m2r[y0:y1, x0:x1]
            ov_t = int(np.logical_and(m1_t, m2_t).sum())
            un_t = int(np.logical_or(m1_t, m2_t).sum())
            tile_jacc.append((ov_t / un_t) if un_t > 0 else float("nan"))
            # tile OD corr inside ROI pixels
            roi_flat = roi_t.reshape(-1)
            od1_t = od1[y0:y1, x0:x1].reshape(-1)[roi_flat]
            od2_t = od2[y0:y1, x0:x1].reshape(-1)[roi_flat]
            tile_od_corr.append(pearson_corr(od1_t, od2_t))
            tile_m1_area_frac.append(int(m1_t.sum()) / roi_t_area)
            tile_m2_area_frac.append(int(m2_t.sum()) / roi_t_area)
    tile_mean_j, tile_med_j, tile_std_j = nan_stats(tile_jacc)
    tile_mean_od, tile_med_od, tile_std_od = nan_stats(tile_od_corr)
    tile_area_correlation = pearson_corr(np.asarray(tile_m1_area_frac, float),
                                         np.asarray(tile_m2_area_frac, float))
    return {
        "marker1": marker1,
        "marker2": marker2,
        "marker1_png_path": marker1_png_path,
        "marker2_png_path": marker2_png_path,
        "dab_png_path": dab_png_path,
        "roi_area_px": roi_area,
        "marker1_area_px": m1_area,
        "marker2_area_px": m2_area,
        "overlap_area_px": overlap,
        "marker1_over_marker2_coloc_ratio": m1_over_m2,
        "marker2_over_marker1_coloc_ratio": m2_over_m1,
        "jaccard_in_roi": jaccard,
        "dice_in_roi": dice,
        "global_phi": global_phi,
        "global_od_pearson": global_od_pearson,
        "weighted_jaccard_continuous": weighted_jaccard_continuous,
        "od_i0_marker1": i0_1,
        "od_i0_marker2": i0_2,
        "tile_size_px": ts,
        "tile_count_used": tiles_used,
        "tile_mean_jaccard": tile_mean_j,
        "tile_median_jaccard": tile_med_j,
        "tile_std_jaccard": tile_std_j,
        "tile_mean_od_corr": tile_mean_od,
        "tile_median_od_corr": tile_med_od,
        "tile_std_od_corr": tile_std_od,
        "tile_area_correlation": tile_area_correlation,
    }
# =============================================================================
# 2) FOLDER WRAPPER (MATCH YOUR NAMING WHERE LEADING IDs DIFFER)
# =============================================================================
def iter_pngs(root: str) -> List[str]:
    out = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".png"):
                out.append(os.path.join(r, f))
    out.sort()
    return out
def stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]
def extract_leading_id_and_marker(st: str) -> Tuple[Optional[str], Optional[str]]:
    """
    From "12_cd163_4-1-roiCrop_DAB_threshold220" -> ("12","cd163")
    """
    parts = st.split("_")
    if len(parts) < 3:
        return None, None
    return parts[0], parts[1]
def extract_core_key(st: str, threshold_tag_regex: str = r"_threshold\d+$") -> Optional[str]:
    """
    Core key is everything after "<ID>_<marker>_" with optional "_threshold###" removed.
      "12_cd163_4-1-roiCrop_DAB_threshold220" -> "4-1-roiCrop_DAB"
      "12_cd163_4-1-roiCrop_DAB"             -> "4-1-roiCrop_DAB"
    """
    parts = st.split("_")
    if len(parts) < 3:
        return None
    core = "_".join(parts[2:])
    core = re.sub(threshold_tag_regex, "", core)
    return core if core else None
def is_thresholded(st: str, threshold_tag_regex: str = r"_threshold\d+$") -> bool:
    return re.search(threshold_tag_regex, st) is not None
def quantify_ihc_colocalization_folder_wrapper(
    marker1: str,
    marker2: str,
    marker1_root_thresholded: str,
    marker2_root_thresholded: str,
    dab_root_roi: str,
    output_csv_path: str,
    *,
    # same quantification params passed through:
    tile_size: int = 256,
    alpha_threshold: int = 1,
    allow_size_mismatch: bool = False,
    od_eps: float = 1.0,
    i0_percentile: float = 95.0,
    i0_fallback: float = 255.0,
    # filename matching params:
    threshold_tag_regex: str = r"_threshold\d+$",
    prefer_dab_from_marker1_id: bool = True,
) -> None:
    """
    Batch wrapper:
      - drives by marker1 thresholded images
      - matches marker2 thresholded by core key
      - matches DAB ROI by core key (prefers same leading ID as marker1 if possible)
      - writes one CSV row per matched triplet
    """
    if not os.path.isdir(marker1_root_thresholded):
        raise ValueError(f"Not found: {marker1_root_thresholded}")
    if not os.path.isdir(marker2_root_thresholded):
        raise ValueError(f"Not found: {marker2_root_thresholded}")
    if not os.path.isdir(dab_root_roi):
        raise ValueError(f"Not found: {dab_root_roi}")
    # index marker2 thresholded by key
    marker2_by_key: Dict[str, List[str]] = {}
    for p in iter_pngs(marker2_root_thresholded):
        st = stem(p)
        if not is_thresholded(st, threshold_tag_regex):
            continue
        key = extract_core_key(st, threshold_tag_regex)
        if key:
            marker2_by_key.setdefault(key, []).append(p)
    # index dab ROI (non-threshold) by key
    dab_by_key: Dict[str, List[str]] = {}
    for p in iter_pngs(dab_root_roi):
        st = stem(p)
        if is_thresholded(st, threshold_tag_regex):
            continue
        key = extract_core_key(st, threshold_tag_regex)
        if key:
            dab_by_key.setdefault(key, []).append(p)
    results: List[Dict[str, object]] = []
    skipped_missing = 0
    for m1_path in iter_pngs(marker1_root_thresholded):
        st1 = stem(m1_path)
        if not is_thresholded(st1, threshold_tag_regex):
            continue
        key = extract_core_key(st1, threshold_tag_regex)
        if not key:
            continue
        m1_id, _m1_marker = extract_leading_id_and_marker(st1)
        m2_candidates = marker2_by_key.get(key, [])
        dab_candidates = dab_by_key.get(key, [])
        if not m2_candidates or not dab_candidates:
            skipped_missing += 1
            continue
        # choose marker2 candidate: prefer marker2 token match
        m2_path = None
        for cand in m2_candidates:
            _id, mk = extract_leading_id_and_marker(stem(cand))
            if mk and mk.lower() == marker2.lower():
                m2_path = cand
                break
        if m2_path is None:
            m2_path = m2_candidates[0]
        # choose dab candidate: prefer same leading ID as marker1
        dab_path = None
        if prefer_dab_from_marker1_id and m1_id is not None:
            for cand in dab_candidates:
                did, _mk = extract_leading_id_and_marker(stem(cand))
                if did == m1_id:
                    dab_path = cand
                    break
        if dab_path is None:
            dab_path = dab_candidates[0]
        row = quantify_triplet_core(
            marker1=marker1,
            marker2=marker2,
            marker1_png_path=m1_path,
            marker2_png_path=m2_path,
            dab_png_path=dab_path,
            tile_size=tile_size,
            alpha_threshold=alpha_threshold,
            allow_size_mismatch=allow_size_mismatch,
            od_eps=od_eps,
            i0_percentile=i0_percentile,
            i0_fallback=i0_fallback,
        )
        row["match_key"] = key  # helpful debug/grouping
        results.append(row)
    if not results:
        raise RuntimeError(
            "No matched triplets found.\n"
            f"Skipped due to missing match: {skipped_missing}\n"
            "Check folder roots + filename pattern."
        )
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)) or ".", exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r)
return 
# =============================================================================
# 3) SINGLE TRIPLET DEMO (USES THE SAME CORE)
# =============================================================================
def demo_single_triplet():
    """
    Edit the three paths below to your real files and run this function.
    It uses quantify_triplet_core() directly (same core as batch wrapper).
    """
    marker1 = "cd163"
    marker2 = "f480"
    m1 = r"H:\workspace\260211IHC\threshold220\roiCropAligned\H-DAB\DAB\cd163\12_cd163_4-1-roiCrop_DAB_threshold220.png"
    m2 = r"H:\workspace\260211IHC\threshold220\roiCropAligned\H-DAB\DAB\f480\13_f480_4-1-roiCrop_DAB_threshold220.png"
    dab = r"H:\workspace\260211IHC\test\roiCropAligned\H-DAB\DAB\cd163\12_cd163_4-1-roiCrop_DAB.png"
    res = quantify_triplet_core(
        marker1=marker1,
        marker2=marker2,
        marker1_png_path=m1,
        marker2_png_path=m2,
        dab_png_path=dab,
        tile_size=256,
        alpha_threshold=1,
        allow_size_mismatch=False,
        od_eps=1.0,
        i0_percentile=95.0,
        i0_fallback=255.0,
    )
    # Print a focused subset
    keys = [
        "marker1_over_marker2_coloc_ratio",
        "marker2_over_marker1_coloc_ratio",
        "jaccard_in_roi",
        "dice_in_roi",
        "global_phi",
        "global_od_pearson",
        "weighted_jaccard_continuous",
        "tile_mean_od_corr",
        "tile_std_od_corr",
        "tile_area_correlation",
        "tile_count_used",
    ]
    for k in keys:
        print(f"{k}: {res.get(k)}")
    return
demo_single_triplet()
import numpy as np
print(np, type(np))
if __name__ == "__main__":
    # Run ONE of the following:
    # (A) Single triplet demo:
    #demo_single_triplet()
    # (B) Batch folder wrapper example:
    quantify_ihc_colocalization_folder_wrapper(
         marker1="cd163",
         marker2="f480",
         marker1_root_thresholded=r"H:\workspace\260211IHC\threshold220\roiCropAligned\H-DAB\DAB\cd163",
         marker2_root_thresholded=r"H:\workspace\260211IHC\threshold220\roiCropAligned\H-DAB\DAB\f480",
         dab_root_roi=r"H:\workspace\260211IHC\test\roiCropAligned\H-DAB\DAB\cd163",
         output_csv_path=r"H:\workspace\260211IHC\test\coloc_threshold220\coloc_cd163_vs_f480.csv",
         tile_size=256,
         alpha_threshold=1,
         allow_size_mismatch=False)
quantify_ihc_colocalization_folder_wrapper(
    marker1="mpo",
    marker2="ly6g",
    marker1_root_thresholded=r"H:\workspace\260211IHC\threshold220\roiCropAligned\H-DAB\DAB\mpo",
    marker2_root_thresholded=r"H:\workspace\260211IHC\threshold220\roiCropAligned\H-DAB\DAB\ly6g",
    dab_root_roi=r"H:\workspace\260211IHC\test\roiCropAligned\H-DAB\DAB\ly6g",
    output_csv_path=r"H:\workspace\260211IHC\test\coloc_threshold220\coloc_mpo_vs_ly6g.csv",
    tile_size=256,
    alpha_threshold=1,
    allow_size_mismatch=False)
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
        1: "LN111+IL-10/VEGF",
        3: "LN111+IL-10",
        2: "LN111",
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
outs = summarize_all_markers_layout_like_excel(
    input_csv_path=r"H:\workspace\260211IHC\test\coloc_threshold220\coloc_cd163_vs_f480.csv",
    summary_target="marker1_over_marker2_coloc_ratio",
    output_folder_path=r"H:\workspace\260211IHC\test\coloc_threshold220\summary",
    column_holder="dab_png_path",
    marker_id_location=1,
    mouse_id_location=2,
    mouse_id_min=4,
    mouse_id_max=13,
)
print("\n".join(outs))

