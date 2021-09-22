

import numpy as np
import keras


from data_object import DataGenImage
from keras.preprocessing.image import ImageDataGenerator

def setup_aug(boole):
    if boole:
        aug = ImageDataGenerator(
                                rotation_range=90,
                                zoom_range=0.15,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                shear_range=0.15,
                                horizontal_flip=True,
                                fill_mode="nearest")
    else:
        aug = None
    return aug

def random_augment(image, aug):
    if aug:
        return aug.random_transform(image)
    else:
        return image

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_gene_object, size=(224,224,3), batch_size=4, shuffle=True, split='train', number=0,
                 one_hot_encoding=True, fully_conv=False, classes=2):
        self.data = data_gene_object
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.size = size
        self.aug = setup_aug(split=="train")
        self.list_IDs = self.data.return_fold(split, number)
        self.one_hot_encoding = one_hot_encoding
        self.fully_conv = fully_conv
        self.classes = classes
        if self.shuffle == True:
            indexes = np.arange(len(self.list_IDs))
            np.random.shuffle(indexes)
            self.list_IDs = self.list_IDs[indexes]
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate data
        list_IDs_temp = self.list_IDs[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        
        if self.shuffle == True:
            indexes = np.arange(len(self.list_IDs))
            np.random.shuffle(indexes)
            self.list_IDs = self.list_IDs[indexes]

    def load_sample(self, index):
            x, y = self.data.cropped(index, self.size)
            x = random_augment(x, self.aug)
            return x, y

    def __data_generation(self, list_id):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.size))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, index in enumerate(list_id):
            X[i], y[i] = self.load_sample(str(index))
        if self.one_hot_encoding:
            y = keras.utils.to_categorical(y, num_classes=self.classes)
        if self.fully_conv:
            y = np.expand_dims(y, 1)
            y = np.expand_dims(y, 1)
        return X, y


def main():
    import matplotlib.pylab as plt
    path = "/Users/naylorpeter/tmp/predict_from_umap_cell/patients/comp3"
    labels_path = "/Users/naylorpeter/tmp/predict_from_umap_cell/patients/multi_class.csv"

    dgi = DataGenImage(path, labels_path, "tumour_cells")
    dgi.create_inner_fold(5, 9)
    dg = DataGenerator(dgi, size=(224,224,3), batch_size=4, 
                       shuffle=True, split='validation', number=0,
                       one_hot_encoding=True, classes=10)
    import pdb; pdb.set_trace()

    x, y = dg[0]
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()
