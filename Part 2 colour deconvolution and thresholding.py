import os
import numpy as np
import imagej
import tifffile
import cv2

import pandas as pd
from PIL import Image
from scipy import stats
from skimage import measure

import imagej, scyjava
scyjava.config.add_option('-Xmx6g')

def deconvolve_and_save_channels_imagej(
    rgba_png_path: str,
    nu_output_dir: str,
    dab_output_dir: str,
    ij
):
    """
    H-DAB color deconvolution using Fiji (interactive mode).
    Saves nucleus and DAB directly into separate folders.
    Reattaches original alpha channel after Fiji export.
    """

    os.makedirs(nu_output_dir, exist_ok=True)
    os.makedirs(dab_output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(rgba_png_path))[0]

    nu_path = os.path.join(nu_output_dir, base + "_Nu.png")
    dab_path = os.path.join(dab_output_dir, base + "_DAB.png")

    in_path = rgba_png_path.replace("\\", "/")
    nu_path_fiji = nu_path.replace("\\", "/")
    dab_path_fiji = dab_path.replace("\\", "/")

    macro = f"""
        open("{in_path}");
        run("Colour Deconvolution", "vectors=[H DAB]");

        titles = getList("image.titles");
        if (titles.length == 0) exit("No images found after deconvolution");

        nu_title = "";
        dab_title = "";

        for (i = 0; i < titles.length; i++) {{
            t = toLowerCase(titles[i]);
            if (indexOf(t, toLowerCase("{base}")) >= 0) {{
                if (indexOf(t, "colour_1") >= 0) nu_title = titles[i];
                if (indexOf(t, "colour_2") >= 0) dab_title = titles[i];
            }}
        }}

        if (nu_title == "" || dab_title == "") {{
            for (i = 0; i < titles.length; i++) {{
                t = toLowerCase(titles[i]);
                if (nu_title == "" && indexOf(t, "colour_1") >= 0) nu_title = titles[i];
                if (dab_title == "" && indexOf(t, "colour_2") >= 0) dab_title = titles[i];
            }}
        }}

        if (nu_title == "" || dab_title == "") {{
            print("=== DEBUG: open image titles ===");
            for (i = 0; i < titles.length; i++) print(titles[i]);
            exit("Colour Deconvolution outputs not found");
        }}

        selectWindow(nu_title);
        saveAs("PNG", "{nu_path_fiji}");

        selectWindow(dab_title);
        saveAs("PNG", "{dab_path_fiji}");

        close("*");
    """

    ij.py.run_macro(macro)

    # Restore alpha
    src = cv2.imread(rgba_png_path, cv2.IMREAD_UNCHANGED)
    if src is None or src.shape[2] != 4:
        raise RuntimeError(f"Input is not RGBA: {rgba_png_path}")

    alpha = src[:, :, 3]

    nu = cv2.imread(nu_path, cv2.IMREAD_GRAYSCALE)
    dab = cv2.imread(dab_path, cv2.IMREAD_GRAYSCALE)

    if nu is None or dab is None:
        raise RuntimeError("Fiji output missing")

    nu_rgba = np.dstack([nu, nu, nu, alpha])
    dab_rgba = np.dstack([dab, dab, dab, alpha])

    cv2.imwrite(nu_path, nu_rgba)
    cv2.imwrite(dab_path, dab_rgba)

    return nu_path, dab_path

def batch_deconvolve_folder_tree(input_root: str, output_root: str, ij):
    """
    Folder structure:

    input_root/
        tnf-a/*.png
        cd163/*.png

    output_root/
        Nucleus/tnf-a/*.png
        DAB/tnf-a/*.png
    """

    nucleus_root = os.path.join(output_root, "Nucleus")
    dab_root = os.path.join(output_root, "DAB")

    os.makedirs(nucleus_root, exist_ok=True)
    os.makedirs(dab_root, exist_ok=True)

    for target in sorted(os.listdir(input_root)):
        target_in = os.path.join(input_root, target)
        if not os.path.isdir(target_in):
            continue

        target_nu_out = os.path.join(nucleus_root, target)
        target_dab_out = os.path.join(dab_root, target)

        for fname in sorted(os.listdir(target_in)):
            if not fname.lower().endswith(".png"):
                continue

            in_path = os.path.join(target_in, fname)

            print(f"[RUN] {in_path}")

            deconvolve_and_save_channels_imagej(
                rgba_png_path=in_path,
                nu_output_dir=target_nu_out,
                dab_output_dir=target_dab_out,
                ij=ij
            )

    print("DONE.")
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
    input_root = _select_directory(
        "Part 2 - Select INPUT root folder containing marker subfolders"
    )
    output_root = _select_directory(
        "Part 2 - Select OUTPUT root folder"
    )

    ij = None
    try:
        ij = imagej.init('sc.fiji:fiji', mode='interactive')
        batch_deconvolve_folder_tree(
            input_root=input_root,
            output_root=output_root,
            ij=ij,
        )
        print(f"\n[INFO] Part 2 finished. Output folder:\n{output_root}")
    finally:
        if ij is not None:
            scyjava.shutdown_jvm()
