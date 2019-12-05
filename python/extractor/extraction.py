import os

from glob import glob
from optparse import OptionParser
from tqdm import tqdm
import pandas as pd 

import skimage.io as io
io.use_plugin('tifffile')


from extract_from_nuclei_prediction import bin_analyser
from feature_object import (PixelSize, MeanIntensity, Centroid,
                            Elongation, Circularity, StdIntensity,
                            MeanIntensityOutsideNuclei, Granulometri,
                            GranulometriOutside, LBP, LBPOutside,
                            ChannelMeanIntensity, ChannelStdIntensity, 
                            ChannelMeanIntensityOutsideNuclei)



coordinates = ["Centroid_x", "Centroid_y", "BBox_x_min",
               "BBox_y_min", "BBox_x_max", "BBox_y_max"] 

list_f = [PixelSize("Pixel_sum", 0), 
          Granulometri(["Grano_1", "Grano_3", "Grano_5", "Grano_7"], 0, [1, 3, 5, 7]),
          GranulometriOutside(["OutGrano_1", "OutGrano_3", "OutGrano_5", "OutGrano_7"], 0, 
                              [1, 3, 5, 7], pixel_marge=10),
          MeanIntensity("Intensity_mean_0", 0), 
          ChannelMeanIntensity(["Channel_Intensity_mean_0{}".format(el) for el in range(3)], 0),
          MeanIntensityOutsideNuclei("Intensity_Outside_nuclei_0", 0,  pixel_marge=10),
          ChannelMeanIntensityOutsideNuclei(["Channel_Intensity_Outside_nuclei_0{}".format(el) for el in range(3)], 0,  pixel_marge=10),
          StdIntensity("Intensity_std_0", 0),
          ChannelStdIntensity(["Channel_Intensity_std_0_c{}".format(el) for el in range(3)], 0),
          Elongation("Elongation", 0),
          Circularity("Circularity", 0),
          Centroid(coordinates, 0)]
          #MeanIntensity("Intensity_mean_5", 5), 
          #MeanIntensity("Intensity_mean_10", 10), 
          # LBP(["It will be renamed to something cool.."], 0),
          # LBPOutside(["It will be renamed to something cool.."], 0,  pixel_marge=10),
          
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

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--bin_folder", dest="bin", type="string",
                      help="bin folder name")
    parser.add_option("--rgb_folder", dest="rgb", type="string",
                      help="rgb folder name")
    parser.add_option("--output_tiles", dest="output_tiles", type="string",
                      help="output table name")
    parser.add_option("--marge", dest="marge", type="int",
                      help="how much to reduce the image size")
    parser.add_option("--name", dest="name", type="str",
                      help="output name for the cell table")
    (options, args) = parser.parse_args()

    pattern = os.path.join(options.rgb, "*.tif")
    last_index = 0
    table_list = []
    mark_cell = options.output_tiles is not None
    if mark_cell:
        check_or_create(options.output_tiles)

    for rgb_path in tqdm(glob(pattern)):
        bin_path = rgb_path.replace("rgb", "bin")

        rgb = io.imread(rgb_path)
        bin_ = io.imread(bin_path)

        ## to ensure each cell as a unique id
        if not nothing(bin_):

            # set centroid to image
            bn = os.path.basename(rgb_path)
            x_, y_ = int(bn.split('_')[1]), int(bn.split('_')[2])
            list_f[-1].set_shift((x_, y_))

            table, labeled = bin_analyser(rgb, bin_, list_f, options.marge, pandas_table=True)
            n = table.shape[0]

            table["index"] = range(last_index, n + last_index)
            table.set_index(["index"], inplace=True)

            last_index += n
            table_list.append(table)
            if mark_cell:
                rgb_marked = mark_inside_cells(rgb, table, 
                                               shift_x=x_,
                                               shift_y=y_)
                io.imsave(bin_path.replace("tiles_bin", options.output_tiles), rgb_marked)
                

    res = pd.concat(table_list, axis=0)
    res = res[(res.T != 0).any()] # drop rows where that are only 0! :) 
    res.to_csv(options.name + ".csv")
