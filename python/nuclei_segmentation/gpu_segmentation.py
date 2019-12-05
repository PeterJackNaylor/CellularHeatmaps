
import os
import numpy as np
import h5py

from tqdm import tqdm
from skimage.io import imsave

from model_creation import create_model
from data_handler import get_input, get_image_from_slide



def predict_each_element(output, slide, parameter_file, model):
    """
    Takes a list of parameters associated to a tiff file.
    From this it extract patches and puts them into a list
    Parameters
    ----------
    output: string
        File output folder name
    slide: string
        Slide name, raw data
    parameter_list : list
        List of lists containing 5 fields, x_0, y_0, level, width, height to crop slide.
    model: keras model to predict segmentation

    Returns
    -------
    collection of segmentation results : hdf5 file
        HDF5 file containing the raw data and segmentation for each tissue.
    Raises
    ------
    """
    try:
        os.mkdir(output)
    except:
        pass
    # size_annot is 2012 - 2 * overlap, where overlap = marge + 92
    # size tile is 2012
    name = os.path.join(output, "probability_{}_{}_{}_{}.h5")
    with open(parameter_file) as f:
        lines = f.readlines()
        print('NOT FULL'); lines = lines[0:2]
        for line in tqdm(lines):
            inputs = get_input(line)
            img = get_image_from_slide(slide, inputs)
            inputs = inputs[0:4]
            prob = model.predict(img)['probability']
            #removing unet padding
            inputs = [inputs[0] + 92, inputs[1] + 92, inputs[2] - 2 * 92, inputs[3] - 2 * 92]

            h5f = h5py.File(name.format(*inputs), 'w')
            h5f.create_dataset('distance', data=prob[:, :, 0])
            h5f.create_dataset('raw', data=img[92:-92, 92:-92])       
            h5f.close()
    f.close()



def main():
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--slide", dest="slide", type="string",
                      help="slide name")
    parser.add_option("--parameter", dest="p", type="string",
                      help="parameter file")
    parser.add_option("--output", dest="output", type="string",
                      help="outputs name")
    parser.add_option("--mean_file", dest="mean_file", type="string",
                      help="mean_file for the model")
    parser.add_option("--log", dest="log", type="string",
                      help="log or weights for the model with feat being the after _")
    (options, _) = parser.parse_args() 

    feat = int(options.log.split('_')[-1])

    model = create_model(options.log, np.load(options.mean_file), feat)
    predict_each_element(options.output, options.slide, options.p, model)

if __name__ == '__main__':
    main()

