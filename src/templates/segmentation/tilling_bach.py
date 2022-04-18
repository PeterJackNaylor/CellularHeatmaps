#!/usr/bin/env python
"""
Input variables:
    - sample: path of a tif WSI image.
    - model: path of the tissue segmentation file.
Output files:
    - {sample}_mask.npy
"""
import os

import numpy as np
from glob import glob
# from numpy.core.numeric import outer
from skimage.transform import resize
from skimage import io

from utils import load_model_v2, get_backbone_model # setup_data, 
from validation import load_meta # is preppended in the makefile


# from useful_plot import coloring_bin, apply_mask_with_highlighted_borders
from utils import setup_data


def main():
    sizex, sizey = int("${size_x}"), int("${size_y}")
    # Load sample
    files = glob("${sample}/*.tif")
    slides = np.zeros((len(files), sizex, sizey, 3), dtype="float32")
    for i, f in enumerate(files):
        slides[i] = io.imread(f)

    # Load segmentation_model
    opt = type('', (), {})()
    opt.meta = os.path.join("${model}", "meta.pkl")
    opt.backbone, opt.model = get_backbone_model(os.path.join("${model}", "final_score.csv"))
    opt.weights = os.path.join("${model}", "model_weights.h5")
    opt.mean, opt.std = load_meta(opt.meta)

    model = load_model_v2(opt.backbone, opt.model, opt.weights)

    ds = setup_data(slides, opt.mean, opt.std, opt.backbone, batch_size=1, image_size=(sizex, sizey))
    res = model.predict(ds)

    np.savez("segmented_tiles.npz", tiles=res, raw=slides)


if __name__ == "__main__":
    main()
