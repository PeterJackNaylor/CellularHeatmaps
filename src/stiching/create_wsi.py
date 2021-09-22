import pdb
import os
import sys
import openslide
from useful_wsi import open_image
#import gi
#gi.require_version('Vips', '8.0')
#from gi.repository import Vips
import pyvips
from os.path import basename, join
from optparse import OptionParser
from glob import glob
from tqdm import tqdm
from skimage.io import imread

def get_options():
    parser = OptionParser()
    parser.add_option("--input", dest="input",type="string",
                    help="input folder name with tif files")
    parser.add_option("--output", dest="output",type="string",
                    help="output name for wsi")
    parser.add_option("--marge", dest="marge", type="int",
                    help="how much to reduce the image size")
    parser.add_option("--slide", dest="slide",type="string",
                    help="slide to get dimensions of wsi")
    (options, _) = parser.parse_args()
    
    return options


if __name__ == "__main__":
    options = get_options()
    margin = options.marge

    basename = os.path.basename(options.slide).split('.')[0]


    # output wsi
    size_x, size_y = open_image(options.slide).dimensions
    img = pyvips.Image.black(size_x, size_y)

    # files to iterate over
    files = join(options.input, "*.tif")

    for f in tqdm(glob(files)):

        tile = pyvips.Image.new_from_file(f, 
                                    access=pyvips.Access.SEQUENTIAL)
        _, _x, _y, _size_x, level = f.split('/')[-1].split('.')[0].split('_')
        _size_y = _size_x

        _size_x, _size_y = int(_size_x) - 2 * margin, int(_size_y) - 2 * margin
        sub_tile = tile.extract_area(margin, margin, _size_x, _size_y)
        img = img.insert(sub_tile, int(_x) + margin, int(_y) + margin)

    img.tiffsave(options.output, compression="jpeg", tile=True, pyramid=True, bigtiff=True)
