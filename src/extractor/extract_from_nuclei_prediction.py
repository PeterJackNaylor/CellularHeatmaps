
import numpy as np
import pandas as pd

from skimage.morphology import watershed, dilation, disk, reconstruction
from skimage.transform import resize
from skimage.measure import regionprops, label
from tqdm import trange, tqdm
from joblib import Parallel, delayed
import time
import asyncio

def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)


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
        time1 = time.time()
        seed = np.zeros_like(bin)
        seed[marge:-marge, marge:-marge] = 1
        mask = bin.copy()
        mask[mask > 0] = 1
        mask[marge:-marge, marge:-marge] = 1
        time2 = time.time()
        reconstructed = reconstruction(seed, mask, 'dilation')
        bin[reconstructed == 0] = 0
        time3 = time.time()
        seed = np.ones_like(bin)
        seed[marge:-marge, marge:-marge] = 0
        mask = bin.copy()
        mask[mask > 0] = 1
        mask[seed > 0] = 1
        time4 = time.time()
        reconstructed = reconstruction(seed, mask, 'dilation')
        frontier = bin.copy()
        time5 = time.time()
        frontier[reconstructed == 0] = 0
        front_lb = label(frontier)
        front_obj = regionprops(front_lb)
        to_remove = np.zeros_like(bin)
        time6 = time.time()
        for obj in front_obj:
            x, y = obj.centroid
            if not (marge < x and x < (bin.shape[0] - marge)) or not (marge < y and y < (bin.shape[1] - marge)):
                lb = obj.label
                to_remove[front_lb == lb] = 1
        time7 = time.time()
        print(f"clear_marge Step1 {(time2 - time1)*1000} ms")
        print(f"clear_marge Step2 {(time3 - time2)*1000} ms")
        print(f"clear_marge Step3 {(time4 - time3)*1000} ms")
        print(f"clear_marge Step4 {(time5 - time4)*1000} ms")
        print(f"clear_marge Step5 {(time6 - time5)*1000} ms")
        print(f"clear_marge Step6 {(time7 - time6)*1000} ms")
        bin[to_remove > 0] = 0
    return bin

import time

def needed_grown_region(list_feature):
    """
    Looks if any of the features needs a specific growing of the objects by dilation.
    """
    res = []
    for feat in list_feature:
        if feat._return_n_extension() not in res:
            res += [feat._return_n_extension()]
    return res


def bin_analyser(rgb_image, bin_image, list_feature, 
                 marge=None, pandas_table=False, do_label=True):
    """
    for each object in the bin image (in the margin), for each feature in list_feature
    bin_analyser returns a table (maybe pandas) where each line corresponds to a object 
    and has many features.

    """
    time1 = time.time()
    bin_image_copy = bin_image.copy()

    p = 0
    for feat in list_feature:
        p += feat.size
    # import pdb; pdb.set_trace()
    bin_image_copy = clear_marge(bin_image_copy, marge)
    time12 = time.time()
    if do_label:
        bin_image_copy = label(bin_image_copy)
    
    time13 = time.time()
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
    
    time2 = time.time()
    grow_region_n = needed_grown_region(list_feature)
    img = {0: bin_image_copy}
    region_prop = {0: regionprops(bin_image_copy)}
    for val in grow_region_n:
        if val != 0:
            img[val] = grow_region(bin_image_copy, val)
            region_prop[val] = regionprops(img[val])
    time3 = time.time()
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
            x_m, y_m, x_M, y_M = tmp_regionprop.bbox
            RGB = rgb_image[x_m:x_M,y_m:y_M]
            BIN = bin_image[x_m:x_M,y_m:y_M]
            table[i, (j + offset_all):(j + offset_all + off_tmp)] = feat._apply_region(RGB, BIN, tmp_regionprop)
            offset_all += feat.size - 1
    time4 = time.time()
    print("decomposition of time:")
    print(f"Step1.0 {(time12 - time1)*1000} ms")
    print(f"Step1.1 {(time13 - time12)*1000} ms")
    print(f"Step1.2 {(time2 - time13)*1000} ms")
    print(f"Step2 {(time3 - time2)*1000} ms")
    print(f"Step3 {(time4 - time3)*1000} ms")
    if pandas_table:
        names = get_names(list_feature)
        return pd.DataFrame(table, columns=names), bin_image_copy
    else:
        return table, bin_image_copy



def needed_grown_region_dic(list_feature):
    """
    Looks if any of the features needs a specific growing of the objects by dilation.
    """
    res = {}
    for feat in list_feature:
        res_apply = feat._return_n_extension()
        if res_apply not in res.keys():
            res[res_apply] = [feat]
        else:
            res[res_apply] += [feat]
    return res

def grow_region(bin_image, n_pix):
    """
    Grows a region to fix size.
    """
    op = disk(n_pix)
    dilated_mask = dilation(bin_image, selem=op)
    return  watershed(dilated_mask, bin_image, mask = dilated_mask)

def dilate(bin, n_pix):
    op = disk(n_pix)
    dilated_mask = dilation(bin, selem=op)
    return dilated_mask


def check_within_margin(rgb_image, marge, cell_prop):
    max_x, max_y = rgb_image.shape[0:2]
    x_c, y_c = cell_prop.centroid
    x_bounds = marge < x_c and x_c < max_x - marge
    y_bounds = marge < y_c and y_c < max_y - marge
    if x_bounds and y_bounds:
        return True
    else:
        return False

def get_crop(rgb_image, bin_image, cell_prop, d=0):
    # d is the dilation resolution to pad to the image 
    x_m, y_m, x_M, y_M = cell_prop.bbox
    r_rgb = rgb_image[(x_m-d):(x_M+d), (y_m-d):(y_M+d)]
    r_bin = bin_image[(x_m-d):(x_M+d), (y_m-d):(y_M+d)].copy()
    r_bin[r_bin != cell_prop.label] = 0
    r_bin[r_bin == cell_prop.label] = 1
    if d > 0:
        r_bin = dilate(r_bin, d)
    return r_rgb, r_bin
    


def get_names_dic(feat_list):
    """
    feat_list: list of feature defined in 'feature_object'
    Returns list of the feature names.
    """
    names = []
    for dilation_res in feat_list.keys():
        for el in feat_list[dilation_res]:
            if el.size != 1:
                for it in range(el.size):
                    names.append(el._return_name()[it])
            else:
                names.append(el._return_name())
    return names


def analyse_cell(cell, rgb_image, marge, p, features_grow_region_n,
                bin_image_copy):
    c_array = np.zeros(shape=p)
    if check_within_margin(rgb_image, marge, cell):
        offset_all = 0 
        for dilation_res in features_grow_region_n.keys():
            rgb_c, bin_c = get_crop(rgb_image, bin_image_copy, cell, d=dilation_res)
            for j, feat in enumerate(features_grow_region_n[dilation_res]):
                off_tmp = feat.size 
                c_array[(offset_all):(offset_all + off_tmp)] = feat._apply_region(rgb_c, bin_c, cell)
                offset_all += off_tmp
    return c_array

def bin_extractor(rgb_image, bin_image, list_feature, 
                marge=None, pandas_table=False, do_label=True, 
                n_jobs=8, save_cells=True):

    bin_image_copy = bin_image.copy()
    if do_label:
        bin_image_copy = label(bin_image_copy)
    
    p = 0
    for feat in list_feature:
        p += feat.size
    time12 = time.time()
    features_grow_region_n = needed_grown_region_dic(list_feature)
    
    def task(cell):
        return analyse_cell(cell, rgb_image, marge, p, features_grow_region_n, bin_image_copy)
    
    cell_descriptors = []
    cell_list = regionprops(bin_image_copy)

    cell_descriptors = Parallel(n_jobs=n_jobs)(delayed(task)(i) for i in cell_list)
    # for cell in tqdm(cell_list):
    #     cell_descriptors.append(task(cell))

    if cell_descriptors:
        cell_matrix = np.stack(cell_descriptors)
        cell_matrix = cell_matrix[cell_matrix.sum(axis=1) != 0]
    else:
        cell_matrix = np.zeros(shape=(0, p))
    
    if pandas_table:
        names = get_names_dic(features_grow_region_n)
        cell_matrix = pd.DataFrame(cell_matrix, columns=names)
    if save_cells:
        if cell_matrix.shape[0]:
            def task_resize(c):
                if check_within_margin(rgb_image, marge, c):
                    rgb_c, bin_c = get_crop(rgb_image, bin_image_copy, c, d=0)
                    r_rgb = resize(rgb_c, (16,16,3), preserve_range=True).astype('uint8')
                    return r_rgb
                else:
                    return None
            n_0 = cell_matrix.shape[0]
            cell_array = Parallel(n_jobs=n_jobs)(delayed(task_resize)(i) for i in cell_list)
            cell_array = [el for el in cell_array if el is not None]
            cell_array = np.stack(cell_array)
        else:
            cell_array = np.zeros(shape=(0, 16, 16, 3), dtype='uint8')
        return cell_matrix, cell_array


    return cell_matrix