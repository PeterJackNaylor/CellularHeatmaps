import numpy as np
from tqdm import tqdm

from utils import setup_data
from useful_wsi import (get_image, white_percentage,
                        patch_sampling)

def check_for_white(img):
    """
    Function to give to wsi_analyse to filter out images that
    are too white. 
    Parameters
    ----------
    img: numpy array corresponding to an rgb image.
    Returns
    -------
    A bool, indicating to keep or remove the img.
    """
    return white_percentage(img, 220, 0.8)

def wsi_analysis(image, model, size, list_roi, margin, opt):
    """
    Tiles a tissue and encodes each tile.
    Parameters
    ----------
    image: string or wsi image, 
        image to tile.
    model: keras model,
        model to encode each tile.
    list_roi: list of list of ints,
        information to tile correctly image.
    Returns
    -------
    Encoded tiles in matrix form. In row the number of tiles 
    and in columns their respective features.
    """

    n = len(list_roi)
    res = np.zeros(shape=(n, size + margin * 2, size + margin * 2, 1), dtype=float)
    raw = np.zeros(shape=(n, size + margin * 2, size + margin * 2, 3), dtype="uint8")

    for (i, para) in tqdm(enumerate(list_roi), total=n):
        raw[i] = get_image(image, para)
    if n < 200:
        ds = setup_data(raw, opt.mean, opt.std, opt.backbone, batch_size=1, image_size=size + 2 * margin)
        res = model.predict(ds)
    else:
        for i in range(0, n, 200):
            outer_i = n if i + 200 > n else i + 200
            ds = setup_data(raw[i:outer_i], opt.mean, opt.std, opt.backbone, batch_size=1, image_size=size + 2 * margin)
            res[i:outer_i] = model.predict(ds)

    return raw, res


def tile(image, level, mask, size, margin, mask_level=5):
    """
    Loads a folder of numpy array into a dictionnary.
    Parameters
    ----------
    image: string or wsi image, 
        image to tile.
    level: int,
        level to which apply the analysis.
    mask_level: int,
        level to which apply the mask tissue segmentation.
    Returns
    -------
    A list of parameters corresponding to the tiles in image.
    """
    def load_gt(ignore):
        return mask
    ## Options regarding the mask creationg, which level to apply the function.
    options_applying_mask = {'mask_level': mask_level, 'mask_function': load_gt}

    ## Options regarding the sampling. Method, level, size, if overlapping or not.
    ## You can even use custom functions. Tolerance for the mask segmentation.
    ## allow overlapping is for when the patch doesn't fit in the image, do you want it?
    ## n_samples and with replacement are for the methods random_patch
    options_sampling = {'sampling_method': "grid", 'analyse_level': level, 
                        'patch_size': (size, size), 'overlapping': margin, 
                        'list_func': [check_for_white], 'mask_tolerance': 0.3,
                        'allow_overlapping': False, 'n_samples': 100, 'with_replacement': False}

    roi_options = dict(options_applying_mask, **options_sampling)

    list_roi = patch_sampling(image, **roi_options)  
    return list_roi

