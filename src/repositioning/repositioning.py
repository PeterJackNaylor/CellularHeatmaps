
import os
import numpy as np
from glob import glob

import skimage as sk
from skimage import morphology as m

from rectpack_utils import place_rectangles

def check_or_create(path):
    """
    If path exists, does nothing otherwise it creates it.
    Parameters
    ----------
    path: string, path for the creation of the folder to check/create
    """
    if not os.path.isdir(path):
        os.makedirs(path)


def get_options():
    import argparse
    parser = argparse.ArgumentParser(
        description='Creating heatmap')
    parser.add_argument('--input', required=True,
                        metavar="str", type=str,
                        help='input file')
    parser.add_argument('--output', required=True,
                        metavar="str", type=str,
                        help='output folder')
    args = parser.parse_args()

    return args

def repositionning(image):
    """
    Repositions the connected components in an image.
    The connected components are defined as the components once we define
    the background as the null values on the third axis.
    Parameters
    ----------
    name: image, 
        Three components
    Returns
    -------
    The same image but with the repositionned components.
    """
    # only the third or 'density' channel
    # test = image.copy().sum(axis=-1)
    test = image[:,:,2].copy()
    # left_push = np.zeros_like(image)
    # left_mask = np.zeros_like(image)[:,:,0]
    # repositioning = np.zeros_like(image)
    test[ test > 0] = 255
    test = m.dilation(test, selem=m.disk(10))
    label = sk.measure.label(test)
    new_image, indiv_rect = place_rectangles(label, image)
    return new_image, indiv_rect


def main():
    options = get_options()
    # num = int(options.comp.split('comp')[-1][0])
    inp = np.load(options.input)
    i_r, mapping = repositionning(inp)
    output = options.output
    check_or_create(output)
    name = os.path.join(output, os.path.basename(options.input).replace('.npy', "r.npy"))

    np.save(name, i_r)
    # beg = "heatmaps_comp{}_repos_".format(num) if options.do_comp else "heatmaps_repos_"
    # folder = "./individual_comp{}/".format(num) if options.do_comp else "./individual/"
    # print(beg + options.slide +  ".npy")
    # np.save(beg + options.slide +  ".npy", i_r)
        
    # try:
    #     os.mkdir(folder)
    # except:
    #     pass
    # for rid in mapping.keys():
    #     sizes, binary = mapping[rid]
    #     x, y, h, w = sizes
    #     sub_npy = i_npy[x:h, y:w]
    #     sub_npy[(1 - binary.astype(int)).astype(bool)] = 0
    #     print(folder + options.slide + "_heatmaps_{}.npy".format(rid))
    #     np.save(folder + options.slide + "_heatmaps_{}.npy".format(rid), sub_npy)


if __name__ == '__main__':
    main()