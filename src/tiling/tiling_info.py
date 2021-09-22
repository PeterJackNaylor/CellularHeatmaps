

# from CuttingPatches import ROI
import os
from numpy import load
from useful_wsi import patch_sampling, open_image, visualise_cut
from optparse import OptionParser
from mask_otsu import make_label_with_otsu

# matplotlib without 
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def CreateFileParam(name, para_list, slidename):
    """
    Creates physically a text file named name where each line as an id 
    and each line as parameters
    Parameters
    ----------
    name : string
        File output name, usually ends in '.txt'
    para_list : list
        List of lists containing 5 fields, x_0, y_0, level, width, height
    slidename: list

    Returns
    -------
    threshold : float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.
    Raises
    ------
    """
    f = open(name, "w")
    line = 1
    for para in para_list:
        pre = "__{}__ ".format(line)
        pre += "{} {} {} {} {}".format(*para)
        pre += " {}".format(slidename)
        pre += "\n"
        f.write(pre)
        line += 1
    f.close()


def get_options():
    parser = OptionParser()

    parser.add_option("--slide", dest="slide", type="string",
                      help="slide name")
    parser.add_option("--output", dest="out", type="string",
                      help="out path")
    parser.add_option("--marge", dest="marge", type="int",
                      help="how much to reduce indexing")
    parser.add_option("--tissue_seg", dest="xml_file", type="str",
                      help="xml file giving the tissue segmentation of the patient tissue")

    (options, _) = parser.parse_args()
    return options

if __name__ == "__main__":
    options = get_options()
    out = options.out + "_tiles.txt"
    out_visu = options.out + "_visualisation.png"

    # because we are doing UNet! :-) this should disappear 
    image = options.slide
    extension = image.split('.')[-1]
    overlap = options.marge + 92  ## because we do segmentation with UNet and we need padding
    slide_file_name = os.path.basename(options.slide).replace(".{}".format(extension), "")


    mask_level = open_image(image).level_count - 2
    def load_gt(img): ## dirty trick to keep it compatible
        lbl = make_label_with_otsu(options.xml_file, img)
        return lbl
    options_applying_mask = {'mask_level': mask_level, 'mask_function': load_gt}

    ## Options regarding the sampling. Method, level, size, if overlapping or not.
    ## You can even use custom functions. Tolerance for the mask segmentation.
    ## allow overlapping is for when the patch doesn't fit in the image, do you want it?
    ## n_samples and with replacement are for the methods random_patch

    # possible size  1964, 1980, 1996, 2012, 2028, 2044]
    #  1500, 1516, 1532, 1548, 1564, 1580, 1596,

    size = 2012 - 2 * overlap
    options_sampling = {'sampling_method': "grid", 'analyse_level': 0, 
                        'patch_size': (size, size), 'overlapping': overlap, 
                        'list_func': [], 'mask_tolerance': 0.1,
                        'allow_overlapping': False}
    roi_options = dict(options_applying_mask, **options_sampling)

    list_roi = patch_sampling(image, **roi_options)  
    CreateFileParam(out, list_roi, slide_file_name)

    ## for tissue check:
    PLOT_ARGS = {'color': 'red', 'size': (12, 12),  'with_show': False,
                 'title': "n_tiles={}".format(len(list_roi))}
    visualise_cut(image, list_roi, res_to_view=mask_level, plot_args=PLOT_ARGS)
    plt.savefig(out_visu)
    # if extension == "tiff":



    ## Options regarding the mask creationg, which level to apply the function.

    # else:
    #     def pred_tissue(img):
    #         from skimage.transform import resize
    #         from segmentation_net import Unet

    #         orig_shape = img.shape[0:2]
    #         img_512 = resize(img, (512, 512), order=1, 
    #                          preserve_range=True, mode='reflect', 
    #                          anti_aliasing=True).astype(img.dtype)
    #         log = os.path.join(options.tissue_model, "Unet__0.001__32")
    #         mean = os.path.join(options.tissue_model, "mean_file.npy")
    #         variables_mod = {
    #             'log' : log,
    #             'n_features' : 32,
    #             'image_size' : (224, 224),
    #             'num_channels' : 3, # because it is rgb
    #             'num_labels' : 2, # because we have two classes: object, background
    #             'fake_batch' : 4,
    #             'mean_array' : load(mean), #mean array of input images
    #         }

    #         model = Unet(**variables_mod)

    #         mask_512 = model.predict(img_512)['predictions'].astype('uint8')
    #         mask_512[mask_512 > 0] = 255
    #         mask = resize(mask_512, orig_shape, order=0, 
    #                       preserve_range=True, mode='reflect', 
    #                       anti_aliasing=True)
    #         return mask
    #     options_applying_mask = {'mask_level': mask_level, 'mask_function': pred_tissue}
