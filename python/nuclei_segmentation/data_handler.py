
import numpy as np

from skimage.transform import resize

from segmentation_net.tf_record import _bytes_feature, _int64_feature
# from Preprocessing.Normalization import PrepNormalizer

from useful_wsi import get_image

def generate_unet_possible(i):
    """
    I was a bit lazy and instead of deriving the formula, I just simulated possible size...
    Parameters
    ----------
    i: integer,
        correspond to width (and height as it is square) of the lowest resolution encoding 3D feature map.
        
    Returns
    -------
    A possible size from integer i.
    """
    def block(j):
        """
        Increase resolution for a convolution block
        """
        return (j+4) * 2
    return block(block(block(block(i)))) + 4

def possible_values(n):
    """
    Generates a list of possible values for unet resolution sizes from a list from 0 to n-1.
    Parameters
    ----------
    n: integer, size
        
    Returns
    -------
    A list of possible values for the unet model.
    """
    x = range(n)
    return list(map(generate_unet_possible, x))

def closest_number(val, num_list):
    """
    Return closest element to val in num_list.
    Parameters
    ----------
    val: integer, float,
        value to find closest element from num_list.
    num_list: list,
        list from which to find the closest element.
    Returns
    -------
    A element of num_list.
    """
    return min(num_list, key=lambda x: abs(x-val))

def get_input(l):
    """
    Slices an input of list_pos (which is a list of list..)
    """
    _, x, y, w, h, l, _ = l.split(' ') #first is the line number, last one is name
    return int(x), int(y), int(w), int(h), int(l)

class resizer_unet:
    """
    Class to deal with the image resizing to correspond to the correst width and height of the unet.
    Parameters
    ----------
    slide: string,
        wsi raw data.
    inputs: list,
        parameter list
    Returns
    -------  
    object with methods for resizing image for unet and resize back
    """
    def __init__(self, slide, inputs):#, NORM):
        """
        Slices crop from the slide, converts it to array and records orginal shape.
        Generates a list to 300 and find the closest shape. (hopefully 300 is big enough)
        """
        image = get_image(slide, inputs)
        image = np.array(image)[:, :, 0:3]
        self.original_image = image
        self.x_orig, self.y_orig = self.original_image.shape[0:2]
        self.possible_unet = possible_values(300)
        self.closest_unet_shape()
        #self.n = NORM

    def prep_image(self, image):
        """
        Preparing the image.
        """
        image = self.preprocess(image)
        return image

    def closest_unet_shape(self):
        """
        Finds closests unet shape.
        """
        self.x_new = closest_number(self.x_orig, self.possible_unet)
        self.y_new = closest_number(self.y_orig, self.possible_unet)
        self.x_lab, self.y_lab = self.x_orig - 92 * 2, self.y_orig - 92 * 2

    def transform_for_analyse(self):
        """
        Transform the image to correct unet size and preps the image.
        """
        img = self.original_image.copy()
        if img.shape[0:2] != (self.x_new, self.y_new):
            img = resize(img, (self.x_new, self.y_new), preserve_range=True).astype(img.dtype)
        img = self.prep_image(img)
        return img

    def transform_back_pred(self, image):
        """
        To transform back.
        """
        if image.shape[0:2] != (self.x_lab, self.y_lab):
            image = resize(image, (self.x_lab, self.y_lab), 
                           order=0, preserve_range=True).astype(image.dtype)
        else:
            print("not resizeing")
        return image

    def preprocess(self, image):
        """
        Here for preprocessing
        """
        # transform = False
        # if image.mean() > 230:
        #     if image.std() < 10:
        #         transform = False
        # if transform:
        #     image = self.n.transform(image)
        return image


def get_image_from_slide(slide, inp):#, n):
    """
    I was a bit lazy and instead of deriving the formula, I just simulated possible size...
    Parameters
    ----------
    slide: string,
        string corresponding to the path to the raw wsi.
    inp: list,
        input parameter corresponding to a list of elements
    model: keras model,
    Returns
    -------
    A the raw image and the segmentation.
    A resized img for segmentation. 
    """
    img_obj = resizer_unet(slide, inp)#, n)
    return img_obj.transform_for_analyse()
