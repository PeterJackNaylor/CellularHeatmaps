import os
import numpy as np

from glob import glob
from optparse import OptionParser
from tqdm import trange
import pandas as pd 

import skimage.io as io
io.use_plugin('tifffile')


from extract_from_nuclei_prediction import bin_analyser, bin_extractor
from feature_object import (PixelSize, MeanIntensity, Centroid,
                            Elongation, Circularity, StdIntensity,
                            Granulometri, LBP,
                            ChannelMeanIntensity, ChannelStdIntensity)



coordinates = ["Centroid_x", "Centroid_y", "Width", "Height", "BBox_x_min",
               "BBox_y_min", "BBox_x_max", "BBox_y_max"] 
list_f = []
for d in [0, 4]:
    list_f.append(PixelSize(f"Pixel_sum_{d}", d))
    list_f.append(MeanIntensity(f"Intensity_mean_{d}", d))
    list_f.append(ChannelMeanIntensity([f"Channel_Intensity_mean_0{el}_{d}".format(el) for el in range(3)], d))
    list_f.append(StdIntensity(f"Intensity_std_{d}", d))
    list_f.append(ChannelStdIntensity([f"Channel_Intensity_std_0_c{el}_{d}".format(el) for el in range(3)], d))
    list_f.append(LBP(["Doesn't matter"], d))
    list_f.append(Granulometri([f"Grano_1_{d}", f"Grano_2_{d}", f"Grano_3_{d}", f"Grano_4_{d}", f"Grano_5_{d}"], d, [1, 2, 3, 4, 5]))

list_f.append(Elongation("Elongation", 0))
list_f.append(Circularity("Circularity", 0))
list_f.append(Centroid(coordinates, 0))
# list_f = [
#         , 
#         PixelSize("Pixel_sum_4", ), 
#         PixelSize("Pixel_sum_8", 10), 
#         ,
#         Granulometri(["Grano_1_5", "Grano_2_5", "Grano_3_5", "Grano_4_5", "Grano_5_5"], 5, [1, 2, 3, 4, 5]),
#         MeanIntensity("Intensity_mean_0", 0), 
#         MeanIntensity("Intensity_mean_5", 5), 
#         # MeanIntensity("Intensity_mean_10", 10), 
#         ChannelMeanIntensity(["Channel_Intensity_mean_0{}_0".format(el) for el in range(3)], 0),
#         ChannelMeanIntensity(["Channel_Intensity_mean_0{}_5".format(el) for el in range(3)], 5),
#         ChannelMeanIntensity(["Channel_Intensity_mean_0{}_10".format(el) for el in range(3)], 10),
#         StdIntensity("Intensity_std_0", 0),
#         StdIntensity("Intensity_std_0", 5),
#         StdIntensity("Intensity_std_0", 10),
#         ChannelStdIntensity(["Channel_Intensity_std_0_c{}_0".format(el) for el in range(3)], 0),
#         ChannelStdIntensity(["Channel_Intensity_std_0_c{}_5".format(el) for el in range(3)], 5),
#         ChannelStdIntensity(["Channel_Intensity_std_0_c{}_10".format(el) for el in range(3)], 10),
#         Elongation("Elongation", 0),
#         Circularity("Circularity", 0),
#         LBP(["Doesn't matter"], 0),
#         LBP(["Doesn't matter"], 5),
#         Centroid(coordinates, 0),
#         # MeanIntensity("Intensity_mean_5", 5), 
#         #MeanIntensity("Intensity_mean_10", 10), 
#           # LBPOutside(["It will be renamed to something cool.."], 0,  pixel_marge=10),
# ]

def check_or_create(path):
    """
    If path exists, does nothing otherwise it creates it.
    Parameters
    ----------
    path: string, path for the creation of the folder to check/create
    """
    if not os.path.isdir(path):
        os.makedirs(path)
def nothing(binary_image):
    """
    check if the image is filled with 0
    """
    return binary_image.sum() == 0

def mark(rgb, line, color=(255, 0, 0), shift_x=0, shift_y=0, length=5, width=1):
    """
    In place algorithm.
    Takes an rgb image and marks a red cross in location (x,y) with an arrow
    whoses branches are of length length and width width.
    """
    x, y = int(line["Centroid_x"]), int(line["Centroid_y"])
    x = x - shift_x
    y = y - shift_y
    rgb[(y-length):(y+length), (x-width):(x+width)] = color
    rgb[(y-width):(y+width), (x-length):(x+length)] = color


def mark_inside_cells(rgb, table, shift_x=0, shift_y=0):
    """
    For every line in table, marks a red cross in rgb.
    """
    new_rgb = rgb.copy()
    table.apply(lambda row: mark(new_rgb, row, shift_x=shift_x, shift_y=shift_y), axis=1)
    return new_rgb

### to remove
import time

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--segmented_batch", dest="segmented_batch", type="string",
                      help="npz object containing the segmentation results and positions")
    parser.add_option("--output_tiles", dest="output_tiles", type="string",
                      help="output table name")
    parser.add_option("--marge", dest="marge", type="int",
                      help="how much to reduce the image size")
    parser.add_option("--name", dest="name", type="str",
                      help="output name for the cell table")
    parser.add_option("-s", "--no_samples",
                  action="store_false", dest="samples", default=True,
                  help="If to save samples")
    parser.add_option("--n_jobs", dest="n_jobs", type="int", default=8,
                      help="Number of jobs")

    (options, args) = parser.parse_args()

    res = np.load(options.segmented_batch)
    rgb = res["raw"]
    bins = res["bins"]
    position = res["positions"]

    name = options.name.split('.')[0]
    n_jobs = int(options.n_jobs)
    last_index = 0
    table_list = []
    mark_cell = options.output_tiles is not None
    save_cells = True
    cell_list = []
    if mark_cell:
        check_or_create(options.output_tiles)

    for i in trange(rgb.shape[0]):

        rgb_ = rgb[i]
        bin_ = bins[i]
        ## to ensure each cell as a unique id
        if not nothing(bin_):

            # set centroid to image
            pos = list(position[i])
            x_, y_ = pos[0:2]
            del pos[3]
            list_f[-1].set_shift((x_, y_))

            start = time.time()
            # table, labeled = bin_analyser(rgb_, bin_, list_f, options.marge, pandas_table=True)
            if save_cells:
                table, cells = bin_extractor(rgb_, bin_, list_f, options.marge, pandas_table=True, save_cells=save_cells, n_jobs=n_jobs)
            else:
                table = bin_extractor(rgb_, bin_, list_f, options.marge, pandas_table=True, save_cells=save_cells, n_jobs=n_jobs)
            # print(f"ONE IMAGE: {(time.time() - start)*1000} ms")
            if table is not None:
                if save_cells:
                    cell_list.append(cells)
                n = table.shape[0]
                table["index"] = range(last_index, n + last_index)
                table.set_index(["index"], inplace=True)

                last_index += n
                table_list.append(table)
                if mark_cell and options.samples:
                    rgb_marked = mark_inside_cells(rgb_, table, 
                                                shift_x=x_,
                                                shift_y=y_)
                    io.imsave(os.path.join(options.output_tiles, "markedcells_{}_{}_{}_{}.tif".format(*pos)), rgb_marked)
                

    res = pd.concat(table_list, axis=0)
    res = res[(res.T != 0).any()] # drop rows where that are only 0! :) 
    res.to_csv(options.name + ".csv")
    if save_cells:
        all_cells = np.vstack(cell_list)
        np.save(options.name + "_tinycells.npy", all_cells)
