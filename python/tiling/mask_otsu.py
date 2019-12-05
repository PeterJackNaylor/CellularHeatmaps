# mask otsu for deriving the patients tissue from an approximate mask.


import os

from math import floor, ceil
from xml.dom import minidom
import numpy as np
from skimage.draw import polygon
from skimage.morphology import dilation
from skimage.color import rgb2gray

from skimage.exposure import histogram
from skimage._shared.utils import warn


def threshold_otsu(image, mask, nbins=256):
    """Return threshold value based on Otsu's method.
    Parameters
    ----------
    image : (N, M) ndarray
        Grayscale input image.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.
    Returns
    -------
    threshold : float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.
    Raises
    ------
    ValueError
         If `image` only contains a single grayscale value.
    References
    ----------
    .. [1] Wikipedia, https://en.wikipedia.org/wiki/Otsu's_Method
    Examples
    --------
    >>> from skimage.data import camera
    >>> image = camera()
    >>> thresh = threshold_otsu(image)
    >>> binary = image <= thresh
    Notes
    -----
    The input image must be grayscale.
    """
    raveled_image = image[mask > 0].ravel()
    if len(image.shape) > 2 and image.shape[-1] in (3, 4):
        msg = "threshold_otsu is expected to work correctly only for " \
              "grayscale images; image shape {0} looks like an RGB image"
        warn(msg.format(image.shape))

    # Check if the image is multi-colored or not
    if raveled_image.min() == raveled_image.max():
        raise ValueError("threshold_otsu is expected to work with images "
                         "having more than one color. The input image seems "
                         "to have just one color {0}.".format(raveled_image.min()))

    hist, bin_centers = histogram(raveled_image, nbins)
    hist = hist.astype(float)
    # class probabilities for all possible thresholds
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of `weight1`/`mean1` should pair with zero values in
    # `weight2`/`mean2`, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return threshold


def closest_int(float_string, maximum):
    f = float(float_string)
    deci = f - floor(f)
    if deci > 0.5:
        res = ceil(f)
    else:
        res = floor(f)
    res = int(res)
    res = min(maximum - 1, max(0, res))
    return res

def make_label(xml_file, rgb_img):
    """ parsers the xml file into a pandas dataframe"""

    row_max, col_max = rgb_img.shape[0:2]
    res = np.zeros((row_max, col_max))

    mydoc = minidom.parse(xml_file)
    polygons = mydoc.getElementsByTagName('polygon')
    for poly in polygons:
        rows = []
        cols = []
        for point in poly.getElementsByTagName('pt'):
            x = int(point.getElementsByTagName('x')[0].firstChild.data)
            y = int(point.getElementsByTagName('y')[0].firstChild.data)
            rows.append(y)
            cols.append(x)
        rr, cc = polygon(rows, cols)
        res[rr, cc] = 1

    return res

to_light_images = ["565330", 
                   "556426",
                   "555029",
                   "541561",
                   "500202", 
                   "485889",
                   "568552"]

def is_change_thresh(xml_file):
    name = os.path.basename(xml_file).split('.')[0]
    change = False
    if name in to_light_images:
        change = True
    return change

def change_thresh(thresh, xml_file):
    print("changing thresh from {}".format(thresh))
    space = (1 - thresh) / 4.
    thresh = thresh + space
    print("to {}".format(thresh))
    return thresh

def make_label_with_otsu(xml_file, rgb_img):
    mask = make_label(xml_file, rgb_img)
    grey_rgb = rgb2gray(rgb_img)
    thresh = threshold_otsu(grey_rgb, mask, nbins=256)
    
    if is_change_thresh(xml_file):
        thresh = change_thresh(thresh, xml_file)
    binary = (grey_rgb < thresh).astype(float)
    merge_mask = mask + binary
    merge_mask[merge_mask != 2] = 0
    merge_mask[merge_mask > 0] = 1
    merge_mask = dilation(merge_mask)
    return merge_mask



def main():
    from skimage.io import imread, imsave
    import matplotlib.pylab as plt
    from skimage.morphology import erosion, disk

    def GetContours(img):
        """
        The image has to be a binary image 
        """
        img[img > 0] = 1
        return dilation(img, disk(2)) - erosion(img, disk(2))
    def apply_contours(img, bin):
        cont = GetContours(bin)
        img[bin > 0] = np.array([0, 0, 0])
        return img

    test = "example/551955.jpg"
    test_xml = "example/551955.xml"

    rgb = imread(test)
    label_mask = make_label_with_otsu(test_xml, rgb)

    _, axes = plt.subplots(ncols=3)
    axes[0].imshow(rgb)
    axes[1].imshow(label_mask)
    axes[2].imshow(apply_contours(rgb, label_mask))
    plt.show()

if __name__ == '__main__':
    main()
