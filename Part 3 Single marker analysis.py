"""
Part III: Single-Marker DAB-Positive Region Quantification Pipeline
This script quantifies DAB-positive regions in thresholded IHC H-DAB images. Thresholded RGBA images define positive regions via alpha > 0, while corresponding DAB ROI images define the full analysis area. It computes positive area, ROI area, percent positivity, grayscale statistics, positive-island morphology, and OD-based metrics including AOD, IOD, and their products with POS_percent. Noticeably, though both registered-but-not-aligned images and registered-and-aligned images need to be thresholded and masked, value-based single-marker quantification should be done employing registered-but-not-aligned (roiRaw) images. Registered-and-aligned images (roiCrop) should be only used for colocalization analysis directly after registered and masked. The script supports recursive folder search, automatic pairing of threshold and DAB images, single-pair demo inspection, and batch CSV export. Filenames follow <slide_index>_<marker>_<sample name>_<roiRaw or roiCrop>_threshold<thresholded value>.png rules, with corresponding DAB images matched accordingly. Dependencies: Python ≥3.7, numpy, opencv-python, Pillow. Author: Zonghan Gan"""
import os
import re
import csv
import math
import numpy as np
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
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
def demo_quantify_threshold_vs_dab(
    threshold_png_path: str,
    dab_png_path: str
) -> dict:
    """
    Demo inspection for one threshold + DAB pair.
    Prints a table including:
      - OD_from_meanI
      - AOD
      - IOD
      - POS_percent (0-100)
      - ODmean_x_POSpercent
      - AOD_x_POSpercent
      - also checks: AOD_x_POSpercent ≈ IOD / ROI_area * 100
    """
    if not os.path.exists(threshold_png_path):
        raise FileNotFoundError(f"Threshold file not found: {threshold_png_path}")
    if not os.path.exists(dab_png_path):
        raise FileNotFoundError(f"DAB file not found: {dab_png_path}")
    thr_rows = quantify_threshold_rgba(threshold_png_path)
    dab_rows = quantify_dab_nontransparent_area(dab_png_path)
    thr = {k: v for k, v in thr_rows}
    dab = {k: v for k, v in dab_rows}
    pos_area = float(thr.get("positive_area_px", 0) or 0)
    roi_area = float(dab.get("dab_nontransparent_area_px", 0) or 0)
    POS_percent = (pos_area / roi_area * 100.0) if roi_area > 0 else None
    OD_from_meanI = thr.get("OD_from_meanI_log10_255_over_meanI", None)
    AOD = thr.get("AOD_mean_OD_pixels", None)
    IOD = thr.get("IOD_sum_OD_pixels", None)
    ODmean_x_POSpercent = (
        float(OD_from_meanI) * float(POS_percent)
        if (OD_from_meanI is not None and POS_percent is not None)
        else None
    )
    AOD_x_POSpercent = (
        float(AOD) * float(POS_percent)
        if (AOD is not None and POS_percent is not None)
        else None
    )
    # Consistency check: AOD*%POS == IOD/ROI*100
    IOD_over_ROI_percent = (
        (float(IOD) / roi_area * 100.0) if (IOD is not None and roi_area > 0) else None
    )
    diff_AOD_vs_IODnorm = (
        (float(AOD_x_POSpercent) - float(IOD_over_ROI_percent))
        if (AOD_x_POSpercent is not None and IOD_over_ROI_percent is not None)
        else None
    )
    combined = {}
    combined.update(thr)
    combined.update(dab)
    combined["POS_percent"] = POS_percent
    combined["ODmean_x_POSpercent"] = ODmean_x_POSpercent
    combined["AOD_x_POSpercent"] = AOD_x_POSpercent
    combined["IOD_over_ROI_percent"] = IOD_over_ROI_percent
    combined["diff_AODxPOS_minus_IODnorm"] = diff_AOD_vs_IODnorm
    # pretty print
    print("\n" + "=" * 70)
    print("DEMO: THRESHOLD vs DAB QUANTIFICATION (OD variants + %POS)")
    print("=" * 70)
    max_key_len = max(len(k) for k in combined.keys())
    for k in sorted(combined.keys()):
        print(f"{k.ljust(max_key_len)} : {combined[k]}")
    print("=" * 70 + "\n")
        return combined
demo_quantify_threshold_vs_dab(
    threshold_png_path=r"H:\workspace\260211IHC\threshold220\roiRaw\H-DAB\DAB\cd163\12_cd163_4-1-roiRaw_DAB_threshold220.png",
    dab_png_path=r"H:\workspace\260211IHC\test\roiRaw\H-DAB\DAB\cd163\12_cd163_4-1-roiRaw_DAB.png"
)
quantify_threshold_vs_dab_to_csv(
    dab_root=r"H:\workspace\260211IHC\test\roiRaw\H-DAB\DAB",
    threshold_root=r"H:\workspace\260211IHC\threshold220\roiRaw\H-DAB\DAB",
    output_folder=r"H:\workspace\260211IHC\test",
    holder1_regex= r"threshold\d{3}\.png",
    out_csv_name="260213_1325_roiRaw_DABsingle_threshold220_quantification.csv",
)
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
outs = summarize_all_markers_layout_like_excel(
    input_csv_path=r"H:\workspace\260211IHC\test\260213_1325_roiRaw_DABsingle_threshold220_quantification.csv",
    summary_target="AOD_x_POSpercent",
    output_folder_path=r"H:\workspace\260211IHC\test\260213_1325_roiRaw_DABsingle_threshold220",
    column_holder="dab_path",
    marker_id_location=1,
    mouse_id_location=2,
    mouse_id_min=4,
    mouse_id_max=13,
)
print("\n".join(outs))
