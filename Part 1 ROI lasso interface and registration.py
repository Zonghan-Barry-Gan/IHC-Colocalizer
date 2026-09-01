"""
IHCs Colocalizer is a Python-based program designed for quantitative colocalization analysis of serial immunohistochemistry (IHC) images from the same biological sample. Although multiplex histochemistry and histofluorescence techniques exist, practical limitations such as chemical orthogonality, spectral leaking, and steric hindrance still constrain multiplex histo-analysis, especially for staining binding markers/proteins simultaneously. Additionally, a single IHC slide is typically 3–5 µm thick, while cell thickness ranges from 10–30 µm. Adjacent slides can therefore be considered as representing a single cell along the z-axis, providing reliable colocalization information for two proteins without methodological interference. The IHCs Colocalizer processes large pathology PNG images, projects and crops matched regions of interest (ROIs) across serial sections, aligns ROIs to near pixel-level accuracy, performs H-DAB color deconvolution, generates DAB-positive threshold masks, quantifies single-marker staining intensity, and evaluates dual-marker spatial colocalization using aligned adjacent sections. The program preserves ROI geometry through RGBA transparency, records image scale transformation, exports intermediate image products and running logs, and produces CSV outputs of single marker quantification and double marker colocalization.
Part I — ROI lasso interface and ROI registration & alignment for IHC series
Part II — IHC H-DAB Channel Deconvolution and positive thresholded masking
Part III — Single-Marker DAB Quantification
Part IV — Dual-Marker IHC Colocalization Analysis
"""
"""
Part I: ROI Lasso Interface and ROI Registration & Alignment for Serial IHC Sections from the Same Sample
This script processes large PNG images of IHC series from one sample in one folder. It provides an interface to lasso ROI on a reference slide, then registers/projects the ROI angular points’ coordinates to all slides. After same ROI is projected to every slides in the series, it first crops ROI registered-but-not-aligned region for all slides as roiRaw, then exportes registered-and-aligned images namely as roiCrop based on ROI homography (4-point exact homography method). It also records the um_per_pixel shift of every images after ROI registration and alignment in csv. The generated scale.csv contains two columns, png_name and um_per_pixel, covering original input images, roiRaw outputs, and roiCrop outputs. Besides, in outcome images, the surrounding blank fillings are annotated as transparent in RGBA PNG, to withhold ROI boundary geometry. Author: Zonghan Gan""""
import os
import re
import csv
import shutil
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import pyvips
from roifile import ImagejRoi
from valis import registration
SUPPORTED_EXTS = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")
# ============================================================
# Utilities
# ============================================================
def extract_mouse_id_from_filename(fname: str) -> str:
    base = os.path.splitext(os.path.basename(fname))[0]
    if "_" not in base:
        raise ValueError(f"Filename has no '_' to extract mouse id: {fname}")
    return base.rsplit("_", 1)[-1].strip()
def sanitize_folder_name(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", name)
def _basename_no_ext(path_or_name: str) -> str:
    return os.path.splitext(os.path.basename(path_or_name))[0]
def _ensure_uint8_rgb_vips(im: pyvips.Image) -> pyvips.Image:
    if im.format != "uchar":
        im = im.cast("uchar")
    if im.bands == 1:
        im = im.bandjoin([im, im]).bandjoin([im])  # 3 bands
    elif im.bands == 2:
        im = im.bandjoin([im[0]])
    elif im.bands >= 3:
        im = im[:3]
    return im
def read_region_rgb_vips(img_path: str, x: int, y: int, w: int, h: int) -> np.ndarray:
    im = pyvips.Image.new_from_file(img_path, access="sequential")
    im = _ensure_uint8_rgb_vips(im)
    reg = im.crop(int(x), int(y), int(w), int(h))
    mem = reg.write_to_memory()
    arr = np.frombuffer(mem, dtype=np.uint8).reshape(reg.height, reg.width, reg.bands)
    return arr
def read_downsample_for_display(img_path: str, disp_scale: int) -> np.ndarray:
    im = pyvips.Image.new_from_file(img_path, access="sequential")
    im = _ensure_uint8_rgb_vips(im)
    out_w = max(1, int(round(im.width / disp_scale)))
    out_h = max(1, int(round(im.height / disp_scale)))
    thumb = im.thumbnail_image(out_w, height=out_h)
    thumb = _ensure_uint8_rgb_vips(thumb)
    mem = thumb.write_to_memory()
    arr = np.frombuffer(mem, dtype=np.uint8).reshape(thumb.height, thumb.width, thumb.bands)
    return arr
def read_imagej_roi_points(roi_path: str) -> np.ndarray:
    """
    Read ImageJ .roi polygon points from disk and return (N,2) float32 in pixel coords.
    """
    if roi_path is None:
        raise ValueError("roi_path is None")
    if not os.path.isfile(roi_path):
        raise FileNotFoundError(f"ROI file not found: {roi_path}")
    roi = ImagejRoi.fromfile(roi_path)
    pts = roi.coordinates()  # (N,2) in (x,y)
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
        raise RuntimeError(f"ROI is not a valid polygon: {roi_path}")
    return pts
# ============================================================
# Evenly sample points along polygon boundary (perimeter)
# ============================================================
def sample_polygon_points_evenly(pts_xy: np.ndarray, k: int) -> np.ndarray:
    """
    Sample k points evenly along polygon boundary. endpoint excluded to avoid duplication.
    """
    pts = np.asarray(pts_xy, dtype=np.float64)
    n = pts.shape[0]
    if n < 2:
        raise RuntimeError("Need at least 2 points to sample boundary")
    if k < 2:
        raise RuntimeError("k must be >= 2")
    p0 = pts
    p1 = np.vstack([pts[1:], pts[:1]])
    seg = p1 - p0
    seg_len = np.sqrt(np.sum(seg * seg, axis=1))
    perim = float(np.sum(seg_len))
    if perim < 1e-9:
        raise RuntimeError("Degenerate polygon perimeter")
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])  # n+1
    targets = np.linspace(0.0, perim, num=k, endpoint=False)
    out = np.zeros((k, 2), dtype=np.float64)
    si = 0
    for i, t in enumerate(targets):
        while si < n and cum[si + 1] <= t:
            si += 1
        d0 = cum[si]
        L = seg_len[si]
        if L < 1e-12:
            out[i] = p0[si]
        else:
            a = (t - d0) / L
            out[i] = p0[si] + a * seg[si]
    return out.astype(np.float32)
# ============================================================
# Crop ROI to RGBA (huge-safe) + return ROI vertices in crop space
# ============================================================
def crop_roi_to_rgba_and_vertices(
    img_path: str,
    roi_full_xy: np.ndarray,
    low_scale: int,
    pad_ratio: float = 0.10,
):
    """
    Crop ROI bbox (with padding), downsample by low_scale, create RGBA where alpha=ROI mask.
    Returns rgba (H,W,4), roi_crop_xy (N,2), crop_size (W,H).
    """
    if low_scale <= 0:
        raise ValueError("low_scale must be positive")
    roi_full = np.asarray(roi_full_xy, dtype=np.float32)
    if roi_full.shape[0] < 3:
        raise RuntimeError("ROI must have at least 3 points")
    im = pyvips.Image.new_from_file(img_path, access="sequential")
    W, H = im.width, im.height
    min_x, min_y = roi_full.min(axis=0)
    max_x, max_y = roi_full.max(axis=0)
    bw = max(1.0, float(max_x - min_x))
    bh = max(1.0, float(max_y - min_y))
    x0d = int(np.floor(min_x - bw * pad_ratio))
    y0d = int(np.floor(min_y - bh * pad_ratio))
    x1d = int(np.ceil(max_x + bw * pad_ratio))
    y1d = int(np.ceil(max_y + bh * pad_ratio))
    crop_w = x1d - x0d
    crop_h = y1d - y0d
    if crop_w <= 1 or crop_h <= 1:
        raise RuntimeError("ROI crop region too small/empty")
    # clip read region to image, but keep padded canvas
    x0c = max(0, x0d); y0c = max(0, y0d)
    x1c = min(W, x1d); y1c = min(H, y1d)
    w_c = x1c - x0c
    h_c = y1c - y0c
    if w_c <= 0 or h_c <= 0:
        raise RuntimeError("ROI bbox is completely outside the image")
    reg_rgb = read_region_rgb_vips(img_path, x0c, y0c, w_c, h_c)
    crop_rgb_full = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
    ox = x0c - x0d
    oy = y0c - y0d
    crop_rgb_full[oy:oy + h_c, ox:ox + w_c] = reg_rgb
    out_w = max(1, int(round(crop_w / low_scale)))
    out_h = max(1, int(round(crop_h / low_scale)))
    crop_rgb = cv2.resize(crop_rgb_full, (out_w, out_h), interpolation=cv2.INTER_AREA)
    roi_crop = roi_full.copy()
    roi_crop[:, 0] = (roi_crop[:, 0] - x0d) / float(low_scale)
    roi_crop[:, 1] = (roi_crop[:, 1] - y0d) / float(low_scale)
    roi_crop = roi_crop.astype(np.float32)
    mask = np.zeros((out_h, out_w), dtype=np.uint8)
    cv2.fillPoly(mask, [roi_crop.astype(np.int32)], 255)
    rgba = np.dstack([crop_rgb, mask]).astype(np.uint8)
    return rgba, roi_crop, (out_w, out_h)
# ============================================================
# Homography (4-point exact) + stable point selection
# ============================================================
def _polygon_signed_area(pts: np.ndarray) -> float:
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[0] < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
def _ensure_same_orientation(src_pts: np.ndarray, ref_pts: np.ndarray) -> np.ndarray:
    """
    If polygon orientation differs (CW/CCW), reverse src order.
    """
    src = np.asarray(src_pts, dtype=np.float32)
    ref = np.asarray(ref_pts, dtype=np.float32)
    a = _polygon_signed_area(src)
    ar = _polygon_signed_area(ref)
    if a == 0 or ar == 0:
        return src
    if (a > 0) != (ar > 0):
        return src[::-1].copy()
    return src
def pick_control_points_evenly(roi_crop_xy: np.ndarray, k_even: int) -> np.ndarray:
    return sample_polygon_points_evenly(roi_crop_xy, k_even).astype(np.float32)
def pick_4_point_subset_max_spread(pts: np.ndarray) -> np.ndarray:
    import itertools
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) < 4:
        raise ValueError(f"Need at least 4 points to choose subset, got {len(pts)}")
    best_score = -1.0
    best_idx = None
    for idx in itertools.combinations(range(len(pts)), 4):
        P = pts[list(idx)]
        dsum = 0.0
        for a, b in itertools.combinations(range(4), 2):
            dsum += float(np.linalg.norm(P[a] - P[b]))
        if dsum > best_score:
            best_score = dsum
            best_idx = idx
    return np.array(best_idx, dtype=np.int32)
def estimate_homography_exact_4(src_all: np.ndarray, dst_all: np.ndarray):
    src_all = np.asarray(src_all, dtype=np.float32)
    dst_all = np.asarray(dst_all, dtype=np.float32)
    n = min(len(src_all), len(dst_all))
    src_all = src_all[:n]
    dst_all = dst_all[:n]
    if n < 4:
        raise RuntimeError("Need at least 4 points for homography")
    idx = pick_4_point_subset_max_spread(src_all)
    src4 = src_all[idx].astype(np.float32)
    dst4 = dst_all[idx].astype(np.float32)
    hull = cv2.convexHull(src4.reshape(-1, 1, 2)).reshape(-1, 2)
    if hull.shape[0] < 3:
        raise RuntimeError("Degenerate 4-point set for homography (hull<3)")
    area = float(abs(cv2.contourArea(hull.astype(np.float32))))
    if area < 1e-2:
        raise RuntimeError("Degenerate 4-point set for homography (area too small)")
    H = cv2.getPerspectiveTransform(src4, dst4)  # exact for these 4 points
    return H.astype(np.float32), src4, dst4
def check_homography_anchor_error(H: np.ndarray, src4: np.ndarray, dst4: np.ndarray, tol_px: float = 1e-3):
    src = src4.reshape(-1, 1, 2).astype(np.float32)
    proj = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    err = np.linalg.norm(proj - dst4, axis=1)
    if float(err.max()) > float(tol_px):
        raise RuntimeError(f"Homography anchor error too high: max={err.max():.6g}px, tol={tol_px}px")
    return err
def rough_scale_from_quad(src4: np.ndarray, dst4: np.ndarray) -> float:
    src4 = np.asarray(src4, dtype=np.float64).reshape(-1, 2)
    dst4 = np.asarray(dst4, dtype=np.float64).reshape(-1, 2)
    if src4.shape[0] != 4 or dst4.shape[0] != 4:
        raise ValueError("src4 and dst4 must be (4,2)")
    def mean_edges(P):
        Q = np.vstack([P, P[:1]])
        edges = np.linalg.norm(Q[1:] - Q[:-1], axis=1)
        return float(np.mean(edges))
    ms = mean_edges(src4)
    md = mean_edges(dst4)
    if ms < 1e-9 or md < 1e-9:
        return 1.0
    return md / ms
# ============================================================
# Main pipeline function
# ============================================================
def draw_and_project_roi_with_valis_png_homography(
    img_folder: str,
    output_folder: str,
    reference_name_substring: str,
    low_scale: int,
    um_per_pixel: float,
    disp_scale_mult: int = 8,
    pad_ratio: float = 0.10,
    # Homography controls
    k_even: int = 6,
    homography_anchor_tol_px: float = 1e-3,
    # Reference ROI input (optional)
    reference_roi_path: str | None = None,
):
    """
    Outputs:
      - output_folder/roi/*.roi
      - output_folder/roiRaw/*-roiRaw.png   (all slides first)
      - output_folder/roi-crop/*-roiCrop.png (second pass; per-slide try/except)
      - output_folder/scale.csv with 2 columns:
            png_name, um_per_pixel
    """
    img_folder = os.path.abspath(img_folder)
    output_folder = os.path.abspath(output_folder)
    if um_per_pixel <= 0:
        raise ValueError("um_per_pixel must be positive")
    if low_scale <= 0:
        raise ValueError("low_scale must be positive")
    k_even = int(k_even)
    if k_even < 4:
        raise ValueError("k_even must be >= 4 for homography")
    img_files = sorted([
        os.path.join(img_folder, f)
        for f in os.listdir(img_folder)
        if f.lower().endswith(SUPPORTED_EXTS)
    ])
    if not img_files:
        raise RuntimeError(f"No supported images found in: {img_folder}")
    ref_candidates = [
        f for f in img_files
        if reference_name_substring.lower() in os.path.basename(f).lower()
    ]
    if len(ref_candidates) != 1:
        raise RuntimeError(
            f"Reference image must be unique (found {len(ref_candidates)} matches for '{reference_name_substring}')"
        )
    ref_img_path = ref_candidates[0]
    ref_img_name = os.path.basename(ref_img_path)
    ref_base = _basename_no_ext(ref_img_name)
    roi_folder = os.path.join(output_folder, "roi")
    raw_folder = os.path.join(output_folder, "roiRaw")
    crop_folder = os.path.join(output_folder, "roi-crop")
    os.makedirs(roi_folder, exist_ok=True)
    os.makedirs(raw_folder, exist_ok=True)
    os.makedirs(crop_folder, exist_ok=True)
    base_um_per_px = float(um_per_pixel) * float(low_scale)
    scale_csv_path = os.path.join(output_folder, "scale.csv")
    with open(scale_csv_path, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["png_name", "um_per_pixel"])
        def _write_scale_row(name_or_path: str, umpx: float):
            w.writerow([os.path.basename(name_or_path), float(umpx)])
        # ------------------------------------------------------------
        # Get ROI on reference (interactive OR from .roi)
        # ------------------------------------------------------------
        if reference_roi_path is not None:
            roi_pts_full = read_imagej_roi_points(reference_roi_path).astype(np.float32)
            if roi_pts_full.shape[0] < 3:
                raise RuntimeError("Provided reference ROI has <3 points")
            print(f"[INFO] Using provided reference ROI: {reference_roi_path}")
        else:
            disp_scale = max(1, int(low_scale * disp_scale_mult))
            disp_img = read_downsample_for_display(ref_img_path, disp_scale)
            roi_pts_disp = []
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(disp_img)
            ax.set_title("Left click: add point | Right drag: pan | Scroll: zoom | Enter: finish")
            is_panning = False
            pan_start = (0, 0)
            xlim_start = None
            ylim_start = None
            def on_mouse_press(event):
                nonlocal is_panning, pan_start, xlim_start, ylim_start
                if event.inaxes != ax:
                    return
                if event.button == 1:
                    if event.xdata is None or event.ydata is None:
                        return
                    roi_pts_disp.append((event.xdata, event.ydata))
                    ax.plot(event.xdata, event.ydata, "r.")
                    plt.draw()
                elif event.button == 3:
                    if event.xdata is None or event.ydata is None:
                        return
                    is_panning = True
                    pan_start = (event.xdata, event.ydata)
                    xlim_start = ax.get_xlim()
                    ylim_start = ax.get_ylim()
            def on_mouse_release(event):
                nonlocal is_panning
                is_panning = False
            def on_mouse_move(event):
                if not is_panning or event.inaxes != ax:
                    return
                if event.xdata is None or event.ydata is None:
                    return
                dx = event.xdata - pan_start[0]
                dy = event.ydata - pan_start[1]
                ax.set_xlim(xlim_start[0] - dx, xlim_start[1] - dx)
                ax.set_ylim(ylim_start[0] - dy, ylim_start[1] - dy)
                plt.draw()
            def on_scroll(event):
                base = 1.2
                cur_xlim = ax.get_xlim()
                cur_ylim = ax.get_ylim()
                x, y = event.xdata, event.ydata
                if x is None or y is None:
                    return
                scale = 1 / base if event.button == "up" else base
                new_w = (cur_xlim[1] - cur_xlim[0]) * scale
                new_h = (cur_ylim[1] - cur_ylim[0]) * scale
                relx = (cur_xlim[1] - x) / (cur_xlim[1] - cur_xlim[0])
                rely = (cur_ylim[1] - y) / (cur_ylim[1] - cur_ylim[0])
                ax.set_xlim([x - new_w * (1 - relx), x + new_w * relx])
                ax.set_ylim([y - new_h * (1 - rely), y + new_h * rely])
                plt.draw()
            def on_key_press(event):
                if event.key == "enter":
                    plt.close(fig)
            fig.canvas.mpl_connect("button_press_event", on_mouse_press)
            fig.canvas.mpl_connect("button_release_event", on_mouse_release)
            fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)
            fig.canvas.mpl_connect("scroll_event", on_scroll)
            fig.canvas.mpl_connect("key_press_event", on_key_press)
            plt.show()
            if len(roi_pts_disp) < 3:
                raise RuntimeError("ROI must have at least 3 points")
            roi_pts_full = (np.asarray(roi_pts_disp, dtype=np.float32) * float(disp_scale)).astype(np.float32)
        # Save reference ROI (level-0)
        ImagejRoi.frompoints(roi_pts_full).tofile(os.path.join(roi_folder, ref_base + ".roi"))
        # ------------------------------------------------------------
        # VALIS registration
        # ------------------------------------------------------------
        registrar = registration.Valis(
            img_folder,
            output_folder,
            reference_img_f=ref_img_path,
            align_to_reference=True
        )
        registrar.register()
        ref_obj = registrar.get_slide(ref_img_name)
        # ------------------------------------------------------------
        # PASS 1: Export ALL roiRaw (including reference), record per-slide crop-space ROI
        # ------------------------------------------------------------
        print("[INFO] PASS 1/2: Exporting roiRaw for all slides...")
        per_slide = {}  # slide_name -> dict(roi_crop_xy, raw_png_path)
        ref_roi_crop_xy = None
        ref_size = None
        ref_align_all = None
        # scale rows: include input names once each in PASS 1
        # (A) reference input row
        _write_scale_row(ref_img_name, float(um_per_pixel))
        # (B) reference roiRaw
        ref_raw_png = os.path.join(raw_folder, ref_base + "-roiRaw.png")
        ref_rgba_raw, ref_roi_crop_xy, ref_size = crop_roi_to_rgba_and_vertices(
            img_path=ref_img_path,
            roi_full_xy=roi_pts_full,
            low_scale=low_scale,
            pad_ratio=pad_ratio,
        )
        plt.imsave(ref_raw_png, ref_rgba_raw)
        _write_scale_row(ref_raw_png, base_um_per_px)
        # Prepare reference control points for alignment (used in PASS 2)
        ref_align_all = pick_control_points_evenly(ref_roi_crop_xy, k_even)
        # Debug ROIs for reference
        ImagejRoi.frompoints(ref_roi_crop_xy).tofile(os.path.join(roi_folder, ref_base + "_CROP_SPACE.roi"))
        ImagejRoi.frompoints(ref_align_all).tofile(os.path.join(roi_folder, ref_base + f"_CROP_EVEN_{k_even}.roi"))
        # Save reference projected roi (already saved as ref_base.roi). Also keep for dict
        per_slide[ref_img_name] = {
            "slide_base": ref_base,
            "roi_crop_xy": ref_roi_crop_xy,
            "raw_png_path": ref_raw_png,
        }
        # Other slides: project -> crop -> save roiRaw
        for img_path in img_files:
            slide_name = os.path.basename(img_path)
            if slide_name == ref_img_name:
                continue
            slide_base = _basename_no_ext(slide_name)
            slide_obj = registrar.get_slide(slide_name)
            # input row
            _write_scale_row(slide_name, float(um_per_pixel))
            # project ROI (level-0)
            roi_full = ref_obj.warp_xy_from_to(
                xy=roi_pts_full,
                to_slide_obj=slide_obj,
                src_slide_level=0,
                src_pt_level=0,
                dst_slide_level=0,
                non_rigid=True,
            ).astype(np.float32)
            ImagejRoi.frompoints(roi_full).tofile(os.path.join(roi_folder, slide_base + ".roi"))
            # crop -> rgba + roi_crop_xy
            rgba_raw, roi_crop_xy, _ = crop_roi_to_rgba_and_vertices(
                img_path=img_path,
                roi_full_xy=roi_full,
                low_scale=low_scale,
                pad_ratio=pad_ratio,
            )
            raw_png_path = os.path.join(raw_folder, slide_base + "-roiRaw.png")
            plt.imsave(raw_png_path, rgba_raw)
            _write_scale_row(raw_png_path, base_um_per_px)
            per_slide[slide_name] = {
                "slide_base": slide_base,
                "roi_crop_xy": roi_crop_xy,
                "raw_png_path": raw_png_path,
            }
        # ------------------------------------------------------------
        # PASS 2: Export roiCrop (aligned), slide-by-slide try/except
        # ------------------------------------------------------------
        print("[INFO] PASS 2/2: Exporting roiCrop (aligned) for all slides...")
        # Reference roiCrop is just roiRaw copied (no alignment needed)
        ref_crop_png = os.path.join(crop_folder, ref_base + "-roiCrop.png")
        shutil.copyfile(ref_raw_png, ref_crop_png)
        _write_scale_row(ref_crop_png, base_um_per_px)
        # Process all other slides
        for slide_name, info in per_slide.items():
            if slide_name == ref_img_name:
                continue
            slide_base = info["slide_base"]
            raw_png_path = info["raw_png_path"]
            roi_crop_xy = info["roi_crop_xy"]
            crop_png_path = os.path.join(crop_folder, slide_base + "-roiCrop.png")
            try:
                # read raw RGBA back (keeps memory low for many slides)
                rgba_raw = cv2.imread(raw_png_path, cv2.IMREAD_UNCHANGED)
                if rgba_raw is None:
                    raise RuntimeError(f"Failed to read roiRaw PNG: {raw_png_path}")
                if rgba_raw.ndim != 3 or rgba_raw.shape[2] != 4:
                    raise RuntimeError(f"roiRaw PNG is not RGBA (4-channel): {raw_png_path}")
                # OpenCV loads PNG as BGRA; convert to RGBA to preserve true color
                rgba_raw = cv2.cvtColor(rgba_raw, cv2.COLOR_BGRA2RGBA)
                # stabilize polygon orientation vs reference
                roi_crop_xy2 = _ensure_same_orientation(roi_crop_xy, ref_roi_crop_xy)
                # sample control points
                src_all = pick_control_points_evenly(roi_crop_xy2, k_even)
                dst_all = ref_align_all.astype(np.float32)
                # compute homography
                H, src4, dst4 = estimate_homography_exact_4(src_all, dst_all)
                _ = check_homography_anchor_error(H, src4, dst4, tol_px=homography_anchor_tol_px)
                # warp
                warped = cv2.warpPerspective(
                    rgba_raw,
                    H,
                    ref_size,  # (width,height)
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0, 0),
                )
                # save (use plt.imsave to be consistent with your previous pipeline)
                plt.imsave(crop_png_path, warped)
                # debug aligned ROI
                roi_pts = roi_crop_xy2.reshape(-1, 1, 2).astype(np.float32)
                roi_warp_xy = cv2.perspectiveTransform(roi_pts, H).reshape(-1, 2)
                ImagejRoi.frompoints(roi_warp_xy).tofile(os.path.join(roi_folder, slide_base + "_CROP_SPACE_ALIGNED_H.roi"))
                ImagejRoi.frompoints(src4).tofile(os.path.join(roi_folder, slide_base + "_ANCHORS_SRC4.roi"))
                ImagejRoi.frompoints(dst4).tofile(os.path.join(roi_folder, slide_base + "_ANCHORS_DST4.roi"))
                # rough um/px
                scale_ratio = float(rough_scale_from_quad(src4, dst4))
                if scale_ratio <= 0:
                    scale_ratio = 1.0
                um_per_px_rough = base_um_per_px / scale_ratio
                _write_scale_row(crop_png_path, float(um_per_px_rough))
            except Exception as e:
                # FALLBACK: copy roiRaw to roiCrop so pipeline continues and output exists
                print(f"[WARN] Homography failed for {slide_name}. Using roiRaw as roiCrop fallback. Error: {e}")
                try:
                    shutil.copyfile(raw_png_path, crop_png_path)
                except Exception as e2:
                    print(f"[ERROR] Fallback copy also failed for {slide_name}: {e2}")
                    # still keep running; don't crash
                # write scale row with base um/px (no alignment)
                _write_scale_row(crop_png_path, base_um_per_px)
        # Cleanup
        try:
            registration.kill_jvm()
        except Exception:
            pass
        try:
            registrar.cleanup()
        except Exception:
            pass
    print("? Completed: roiRaw exported first, then roiCrop with safe per-slide homography.")
    return output_folder
# ============================================================
# SINGLE-MOUSE runner (NO subfolders in input)
# ============================================================
def run_valis_on_single_mouse_png_homography(
    img_folder: str,
    valis_root: str,
    um_per_pixel: float,
    reference_name_substring: str = "HE",
    low_scale: int = 16,
    disp_scale_mult: int = 8,
    pad_ratio: float = 0.10,
    k_even: int = 6,
    reference_roi_path: str | None = None,
):
    img_folder = os.path.abspath(img_folder)
    valis_root = os.path.abspath(valis_root)
    if um_per_pixel <= 0:
        raise ValueError("um_per_pixel must be positive")
    if not os.path.isdir(img_folder):
        raise ValueError(f"[ERROR] img_folder does not exist: {img_folder}")
    os.makedirs(valis_root, exist_ok=True)
    img_files = sorted([
        f for f in os.listdir(img_folder)
        if f.lower().endswith(SUPPORTED_EXTS)
    ])
    if not img_files:
        raise RuntimeError(f"[ERROR] No supported image files found in {img_folder}")
    mouse_ids = {extract_mouse_id_from_filename(f) for f in img_files}
    if len(mouse_ids) != 1:
        raise RuntimeError(
            f"[ERROR] Expected exactly 1 mouse id in folder, found {len(mouse_ids)}: {sorted(mouse_ids)}"
        )
    mouse_id = next(iter(mouse_ids))
    mouse_id_safe = sanitize_folder_name(mouse_id)
    output_folder = os.path.join(valis_root, mouse_id_safe)
    os.makedirs(output_folder, exist_ok=True)
    print("\n" + "=" * 70)
    print("[INFO] Processing single-mouse folder (HOMOGRAPHY + roiRaw first + safe roiCrop)")
    print(f"[INFO] Mouse id      : {mouse_id}")
    print(f"[INFO] IMG folder    : {img_folder}")
    print(f"[INFO] Output folder : {output_folder}")
    print("=" * 70)
    if not any(reference_name_substring.lower() in f.lower() for f in img_files):
        raise RuntimeError(
            f"[ERROR] No reference image containing '{reference_name_substring}' found in {img_folder}"
        )
    draw_and_project_roi_with_valis_png_homography(
        img_folder=img_folder,
        output_folder=output_folder,
        reference_name_substring=reference_name_substring,
        low_scale=low_scale,
        um_per_pixel=um_per_pixel,
        disp_scale_mult=disp_scale_mult,
        pad_ratio=pad_ratio,
        k_even=k_even,
        reference_roi_path=reference_roi_path,
    )
    print("\n[INFO] Done.")
    return output_folder
# ============================================================
# Merge scale.csv across all outputs under valis_root
# ============================================================
def merge_mouse_scale_csvs(valis_root: str, output_filename_suffix: str = "roi_scale.csv"):
    valis_root = os.path.abspath(valis_root)
    if not os.path.isdir(valis_root):
        raise ValueError(f"[ERROR] valis_root does not exist: {valis_root}")
    subfolders = sorted([
        d for d in os.listdir(valis_root)
        if os.path.isdir(os.path.join(valis_root, d))
    ])
    if not subfolders:
        raise RuntimeError(f"[ERROR] No subfolders found under {valis_root}")
    merged = []
    print(f"[INFO] Searching scale.csv in {len(subfolders)} subfolders")
    for folder in subfolders:
        p = os.path.join(valis_root, folder, "scale.csv")
        if not os.path.isfile(p):
            continue
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[ERROR] Failed to read {p}: {e}")
            continue
        df.insert(0, "mouse_id", folder)
        merged.append(df)
        print(f"[INFO] Loaded scale.csv from {folder}")
    if not merged:
        raise RuntimeError("[ERROR] No valid scale.csv files were loaded.")
    out_df = pd.concat(merged, ignore_index=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(valis_root, f"{timestamp}_{output_filename_suffix}")
    out_df.to_csv(out_csv, index=False)
    print(f"\n[INFO] Merged CSV saved to:\n       {out_csv}")
    return out_csv
# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    # Example:
    # img_folder = r"D:\yourdata\ms3_png"
    # valis_root = r"D:\yourdata\valis_out"
    # um_per_pixel = 0.504   # example; you should pass the correct value
    # reference_roi_path = None
    #   - or reference_roi_path = r"D:\yourdata\valis_out\ms3\roi\HE_ms3-1.roi"
    run_valis_on_single_mouse_png_homography(
        img_folder=r"d:\pngoutput\11-4",
        valis_root=r"d:\valis_out_260210-2",
        um_per_pixel=0.261438,
        reference_name_substring="f480",
        low_scale=1,
        disp_scale_mult=16,
        pad_ratio=0.05,
        k_even=6,  # recommend 4~6
        reference_roi_path=None
        #reference_roi_path=r"F:\valis_out\6-2\roi\13-f480_6-2.roi"
    )
    
     #Merge all mouse outputs under valis_root#
    merge_mouse_scale_csvs(
        valis_root=r"H:\workspace\260211IHC\valis",
        output_filename_suffix="roi_scale.csv"
    )
