
import sys
sys.path.append('..')
import os
from os.path import join

import numpy as np
from glob import glob
import h5py
from tqdm import tqdm 
from scipy.ndimage import binary_fill_holes

import skimage.measure as meas
from skimage import io
from skimage.morphology import remove_small_objects
from skimage.segmentation import mark_boundaries
from skimage import img_as_ubyte
io.use_plugin('tifffile')

from dynamic_watershed.dynamic_watershed import post_process, generate_wsl

def add_contours(image, label, color = (0, 1, 0)):
    """
    Overlays the label ontop of the image with a given colour.
    Parameters
    ----------
    image: rgb image
    label: integer map, where each integer represents a connected component
    color: tuple of three elements corresponding to the percentage of red, green and blue
        
    Returns
    -------
    An image of same size and type of image but with overlaid segmentation.
    """
    res = mark_boundaries(image, label, color=color)
    res = img_as_ubyte(res)
    return res


def check_or_create(path):
    """
    If path exists, does nothing otherwise it creates it.
    Parameters
    ----------
    path: string, path for the creation of the folder to check/create
    """
    if not os.path.isdir(path):
        os.makedirs(path)

def fill_holes(image):
    """
    Fills holes in a image
    Parameters
    ----------
    image: fills holls in binary image with mathematical morphology provided by skimage.
    Returns
    -------
    An image of the same size as the input
    """
    rec = binary_fill_holes(image)
    return rec

def post_process_out(pred, img):
    """
    Main post processing function, applies the post processing defined in the paper:
    Nuclei segmentation in histopathology images using deep neural networks by P. Naylor, M. Lae, F. Reyal and T. Walter.
    Parameters
    ----------
    pred: prediction image of type integer
    img: raw image of anytype
    Returns
    -------
    A tuple of four images (B, C, P, img).
        B: the integer map where each integer is a object,
        C: img overlaid with B,
        P: distance map,
        img: raw input img,
    """
    hp = {'p1': 1, 'p2':0.5}
    scale = 1
    min_size = 128

    pred[pred < 0] = 0.
    pred[pred > 255] = 255.

    labeled_pic = post_process(pred, hp["p1"] * scale, hp["p2"] * scale)
    borders_labeled_pic = generate_wsl(labeled_pic)
    labeled_pic = remove_small_objects(labeled_pic, min_size=min_size)
    labeled_pic[labeled_pic > 0] = 255
    labeled_pic[borders_labeled_pic > 0] = 0
    labeled_pic = fill_holes(labeled_pic)

    img = img.astype('uint8')
    B = labeled_pic.astype('uint8')
    C = add_contours(img, meas.label(labeled_pic))

    pred[pred > 20] = 20
    pred = pred * 255. / 20
    P = pred.astype('uint8')

    return B, C, P, img

def main():
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--input", dest="input", type="string",
                      help="record name")
    (options, _) = parser.parse_args() 

    files = glob(options.input)
    tiles_prob = "./tiles_prob"
    tiles_contours = "./tiles_contours"
    tiles_bin = "./tiles_bin"
    tiles_rgb = "./tiles_rgb"

    folders = [tiles_bin, tiles_contours, tiles_prob, tiles_rgb]
    out_names = [join(f, f.split('_')[-1] + "_{}_{}_{}_{}.tif") for f in folders]

    for f in folders:
        check_or_create(f)


    for f in tqdm(files):
        h5f = h5py.File(f, 'r')
        distance = h5f['distance'][:]
        rgb = h5f['raw'][:]
        h5f.close()
        list_img = post_process_out(distance, rgb)
        inp = os.path.basename(f).split('.')[0].split('_')
        del inp[0]
        for image, name in zip(list_img, out_names):
            io.imsave(name.format(*inp), image, resolution=[1.0, 1.0])

if __name__ == '__main__':
    main()

