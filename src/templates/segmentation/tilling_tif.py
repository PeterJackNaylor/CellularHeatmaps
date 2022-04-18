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
# from numpy.core.numeric import outer
from skimage.transform import resize
from skimage import io

from utils import load_model_v2, get_backbone_model # setup_data, 
from validation import load_meta # is preppended in the makefile

from useful_wsi import open_image, get_whole_image
# get_image, white_percentage, patch_sampling, get_size, get_x_y_from_0,

# from useful_plot import coloring_bin, apply_mask_with_highlighted_borders
from templates.segmentation.tilling_utils import tile, wsi_analysis
from tqdm import tqdm

def main():
    size = int("${size}")
    margin = int("${wsi_margin}")
    # Load sample
    slide = open_image("${sample}")
    mask = io.imread("${mask}")[:,:,0]

    original_size = get_whole_image(slide, slide.level_count - 2).shape
    mask = resize(
            mask,
            original_size[:2],
            preserve_range=True,
            anti_aliasing=True,
            order=0
        )

    # Load segmentation_model
    opt = type('', (), {})()
    opt.meta = os.path.join("${model}", "meta.pkl")
    opt.backbone, opt.model = get_backbone_model(os.path.join("${model}", "final_score.csv"))
    opt.weights = os.path.join("${model}", "model_weights.h5")
    opt.mean, opt.std = load_meta(opt.meta)

    model = load_model_v2(opt.backbone, opt.model, opt.weights)

    # Divide and analyse
    list_positions = tile(slide, 0, mask, size, margin, slide.level_count - 2)
    raw, segmented_tiles = wsi_analysis(slide, model, size, list_positions, margin, opt)
    np.savez("segmented_tiles.npz", tiles=segmented_tiles, positions=list_positions, raw=raw)


if __name__ == "__main__":
    main()
