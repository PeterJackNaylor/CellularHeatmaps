
import numpy as np
from glob import glob
from tqdm import tqdm
import pandas as pd
import os
from skimage.util import crop, pad
from sklearn.model_selection import StratifiedKFold

def t_name(name):
    """
    How to process the file name to get the key.
    """
    basname = os.path.basename(name).split('.')[0]
    # basname = basname.split('_')[-1]
    return basname

def load_folder(path):
    """
    Loads a folder of numpy array into a dictionnary.
    Parameters
    ----------
    path: string, 
        path to folder from which to find 'npy'
    Returns
    -------
    A dictionnary where the key is the tissue id and
    the item is the tissue heatmaps.
    """
    files = glob(path + "/*.npy")
    loaded = {t_name(f): np.load(f) for f in tqdm(files)}
    # tt = [loaded[t_name(f)].shape for f in files]
    # import pdb; pdb.set_trace()
    # checks sizes how they go down
    return loaded

def load_labels(label_path, label_interest):
    """
    Loads the label table.
    Parameters
    ----------
    label_path: string, 
        path to label csv table
    label_interest: string, 
        name of variable to predict. Has to be in the table loaded by
        the parameter label_path.
    Returns
    -------
    A tuple of series where the first is the variable of interest,
    the second is the outer test folds, the third the variable to stratefy 
    the inner folds by.
    """
    table = pd.read_csv(label_path)
    table = table.set_index(['Biopsy'])
    y = table[label_interest]
    stratefied_variable = table[label_interest]
    folds = table["fold"]
    return y, folds, stratefied_variable


def crop_pad_around(image, size):
    """
    Pads or crops an image so that the image is of a given size.
    Parameters
    ----------
    image: numpy array, 
        3 channel numpy array to crop/pad.
    size: tuple of integers, 
        size to achieve.
    Returns
    -------
    A padded or croped version of image so that image
    has a size of size.
    """
    x, y, z = image.shape

    x_pad_width = size[0] - x if size[0] > x else 0
    y_pad_width = size[1] - y if size[1] > y else 0
    if x_pad_width > 0 or y_pad_width > 0:
        pad_width = [(x_pad_width, y_pad_width) for _ in range(2)]
        pad_width +=[(0, 0)]
        image = np.pad(image, pad_width, mode='constant')

    x, y, z = image.shape
    shapes = [x, y, z]
    x_crop_width = x - size[0] if x > size[0] else 0
    y_crop_width = y - size[1] if y > size[1] else 0
    if x_crop_width > 0 or y_crop_width > 0:
        crops = []
        for i, c in enumerate([x_crop_width, y_crop_width]):
            crop_v = np.random.randint(0, c) if c != 0 else 0
            crops.append((crop_v, shapes[i] - size[i] - crop_v))
        crops.append((0,0))
        image = crop(image, crops, copy=True)
    return image

class DataGenImage():
    'Generates data for Keras datagenerator'
    def __init__(self, path, label_file, label_interest, categorize=False, classes=2):
        'Initialization'
        self.mapper = load_folder(path)
        # vector y, fold, and stratefied
        self.y, self.f, self.sv = load_labels(label_file, label_interest)
        self.classes = classes
        self.folds_focus = False
    def __getitem__(self, index):
        """
        Get item, when given a biopsy id returns image to a given size.
        ----------
        index: string, 
            string existing in the mapper dictionnary
        Returns
        -------
        A biopsy heatmap.
        """
        return self.mapper[index], self.y.ix[index]

    def cropped(self, index, size):
        image, label = self.__getitem__(index)
        image = crop_pad_around(image, size)
        return image, label

    def return_keys(self):
        return list(self.mapper.keys())

    def return_weights(self):
        class_weight = {}
        train, val = self.index_folds[0]
        n = self.y.ix[train].shape[0] + self.y.ix[val].shape[0]
        for i in range(self.classes):
            size_i = (self.y.ix[train] == i).astype(int).sum() + (y.labels.ix[val] == i).astype(int).sum()
            class_weight[i] = (1 - size_i / n)
        return class_weight

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.mapper))

    def return_fold(self, split, number):
        if self.folds_focus:
            if split == "train":
                train, val = self.index_folds[number]
                return train
            elif split == "validation":
                train, val = self.index_folds[number]
                return val
            else:
                return self.test_folds
        else:
            print("cant do, need to focus folds with create_inner_fold")

    def create_inner_fold(self, nber_splits, test_fold):
        """
        Creates inner stratefied folds.
        """
        self.test_folds = self.f[self.f == test_fold].index
        for_train = self.f[self.f != test_fold].index

        skf = StratifiedKFold(n_splits=nber_splits, shuffle=True)

        stratefied_variable = self.sv[self.f != test_fold]
        obj = skf.split(for_train, stratefied_variable)

        self.index_folds = [(for_train[train_index], for_train[val_index]) for train_index, val_index in obj]
        self.folds_focus = True

def main():
    import matplotlib.pylab as plt
    path = "/mnt/data3/pnaylor/ProjectFabien/outputs/heat_maps_small_8/comp3"
    labels_path = "/mnt/data3/pnaylor/ProjectFabien/outputs/multi_class.csv"

    dgi = DataGenImage(path, labels_path, "RCB_class")
    index = '500169'
    x, y = dgi.cropped(index, (224, 224, 3))
    dgi.return_fold("validation", 4)
    dgi.create_inner_fold(5, 9)
    dgi.return_fold("validation", 4)
    import pdb; pdb.set_trace()

def test_crop():
    import matplotlib.pylab as plt
    path = "/mnt/data3/pnaylor/ProjectFabien/outputs/heat_maps_small_8/comp3"
    labels_path = "/mnt/data3/pnaylor/ProjectFabien/outputs/multi_class.csv"

    dgi = DataGenImage(path, labels_path, "RCB_class")
    dgi.create_inner_fold(5, 9)
    peaps = dgi.return_fold("train", 4)
    for peap in peaps:
        x, y = dgi.cropped(str(peap), (224, 224, 3))
        print(x.shape)
    peaps = dgi.return_fold("validation", 4)
    for peap in peaps:
        x, y = dgi.cropped(str(peap), (224, 224, 3))
        print(x.shape)
    peaps = dgi.return_fold("test", 4)
    for peap in peaps:
        x, y = dgi.cropped(str(peap), (224, 224, 3))
        print(x.shape)

    import pdb; pdb.set_trace()    

def test_weight():
    import matplotlib.pylab as plt
    path = "/mnt/data3/pnaylor/ProjectFabien/outputs/heat_maps_small_8/comp3"
    labels_path = "/mnt/data3/pnaylor/ProjectFabien/outputs/multi_class.csv"

    dgi = DataGenImage(path, labels_path, "tumour_cells")
    dgi.create_inner_fold(5, 9)
    w = dgi.return_weights()
    print(w)

if __name__ == '__main__':
    # test_weight()
    test_crop()
    main()
