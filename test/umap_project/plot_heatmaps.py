
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import os

from glob import glob
import numpy as np

from skimage.io import imsave

def check_or_create(path):
    """
    If path exists, does nothing otherwise it creates it.
    Parameters
    ----------
    path: string, path for the creation of the folder to check/create
    """
    if not os.path.isdir(path):
        os.makedirs(path)


def options_parser():
    import argparse
    parser = argparse.ArgumentParser(
        description='Creating heatmap')
    parser.add_argument('--input_folder', required=True,
                        metavar="str", type=str,
                        help='input folder')
    parser.add_argument('--output_folder', required=True,
                        metavar="int", type=str,
                        help='output folder')
    args = parser.parse_args()
    return args


def main():
    options = options_parser()

    check_or_create(options.output_folder)

    for f in glob(options.input_folder + "/*.npy"):
        mat = np.load(f)
        mat[mat[:,:,2] == 0] = np.nan
        name = os.path.basename(f).replace('.npy', "_axis{}.pdf")
        for i in range(3):
            plt.imshow(mat[:,:,i], cmap='jet')
            plt.colorbar()
            out = os.path.join(options.output_folder, name).format(i+1)
            plt.axis('off')
            plt.savefig(out, bbox_inches='tight')
            plt.close()



if __name__ == '__main__':
    main()