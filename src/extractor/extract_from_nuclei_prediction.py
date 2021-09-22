
import numpy as np
import pandas as pd

from skimage.morphology import watershed, dilation, disk, reconstruction
from skimage.measure import regionprops, label
from tqdm import trange
from joblib import Parallel, delayed
import time
import asyncio

def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

import multiprocessing

def get_names(feat_list):
    """
    feat_list: list of feature defined in 'feature_object'
    Returns list of the feature names.
    """
    names = []
    for el in feat_list:
        if el.size != 1:
            for it in range(el.size):
                names.append(el._return_name()[it])
        else:
            names.append(el._return_name())
    return names

def clear_marge(bin, marge):
    """
    Removes the object within the margins of a given binary image
    bin: binary image
    marge: integer
    """
    if marge is not None and marge != 0:
        seed = np.zeros_like(bin)
        seed[marge:-marge, marge:-marge] = 1
        mask = bin.copy()
        mask[mask > 0] = 1
        mask[marge:-marge, marge:-marge] = 1
        reconstructed = reconstruction(seed, mask, 'dilation')
        bin[reconstructed == 0] = 0

        seed = np.ones_like(bin)
        seed[marge:-marge, marge:-marge] = 0
        mask = bin.copy()
        mask[mask > 0] = 1
        mask[seed > 0] = 1
        reconstructed = reconstruction(seed, mask, 'dilation')
        frontier = bin.copy()
        frontier[reconstructed == 0] = 0
        front_lb = label(frontier)
        front_obj = regionprops(front_lb)
        to_remove = np.zeros_like(bin)
        for obj in front_obj:
            x, y = obj.centroid
            if not (marge < x and x < (bin.shape[0] - marge)) or not (marge < y and y < (bin.shape[1] - marge)):
                lb = obj.label
                to_remove[front_lb == lb] = 1
        bin[to_remove > 0] = 0
    return bin

         

#    return bin

def bin_analyser(rgb_image, bin_image, list_feature, 
                 marge=None, pandas_table=False, do_label=True):
    """
    for each object in the bin image (in the margin), for each feature in list_feature
    bin_analyser returns a table (maybe pandas) where each line corresponds to a object 
    and has many features.

    """
    bin_image_copy = bin_image.copy()

    p = 0
    for feat in list_feature:
        p += feat.size
    bin_image_copy = clear_marge(bin_image_copy, marge)
    if do_label:
        bin_image_copy = label(bin_image_copy)

    if len(np.unique(bin_image_copy)) != 2:
        if len(np.unique(bin_image_copy)) == 1:
            if 0 in bin_image_copy:
                print("Return blank matrix. Change this shit")
                white_npy = np.zeros(shape=(1, p))
                if not pandas_table:
                    return white_npy, bin_image_copy
                else:
                    names = get_names(list_feature) 
                    return pd.DataFrame(white_npy, columns=names), bin_image_copy
            else:
                print("Error, must give a bin image.")
    
    grow_region_n = needed_grown_region(list_feature)
    img = {0: bin_image_copy}
    region_prop = {0: regionprops(bin_image_copy)}
    for val in grow_region_n:
        if val != 0:
            img[val] = grow_region(bin_image_copy, val)
            region_prop[val] = regionprops(img[val])
    
    n = len(region_prop[0])

    table = np.zeros(shape=(n, p))

    # start = time.time()
    # offsets = []
    # for i in range(n):
    #     offset_i = [0] 
    #     offset_count = 0
    #     for j, feat in enumerate(list_feature):
    #         offset_count += feat.size - 1
    #         offset_i.append(offset_count)
    #     offsets.append(offset_i)

    # @background
    # def process_image(i):
    #     for j, feat in enumerate(list_feature):
    #         off_tmp = feat.size
    #         tmp_regionprop = region_prop[feat._return_n_extension()][i]
    #         table[i, (j + offsets[i][j]):(j + offsets[i][j] + off_tmp)] = feat._apply_region(tmp_regionprop, rgb_image)
    
    # for i in range(n):
    #     process_image(i)
    # Parallel(n_jobs=10)(delayed(process_image)(i) for i in range(n))

    # pool = multiprocessing.Pool(10)
    # t = pool.map(process_image, range(0, n))

    for i in range(n):
        offset_all = 0 
        for j, feat in enumerate(list_feature):
            off_tmp = feat.size   
            tmp_regionprop = region_prop[feat._return_n_extension()][i]
            table[i, (j + offset_all):(j + offset_all + off_tmp)] = feat._apply_region(tmp_regionprop, rgb_image)
            offset_all += feat.size - 1

    if pandas_table:
        names = get_names(list_feature)
        return pd.DataFrame(table, columns=names), bin_image_copy
    else:
        return table, bin_image_copy

def needed_grown_region(list_feature):
    """
    Looks if any of the features needs a specific growing of the objects by dilation.
    """
    res = []
    for feat in list_feature:
        if feat._return_n_extension() not in res:
            res += [feat._return_n_extension()]
    return res

def grow_region(bin_image, n_pix):
    """
    Grows a region to fix size.
    """
    op = disk(n_pix)
    dilated_mask = dilation(bin_image, selem=op)
    return  watershed(dilated_mask, bin_image, mask = dilated_mask)
