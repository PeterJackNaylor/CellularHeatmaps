
import os
import pickle
import numpy as np
from tqdm import tqdm

from useful_wsi import open_image, get_whole_image
from train_umap import normalise_csv, drop_na_axis


def options_parser():
    import argparse
    parser = argparse.ArgumentParser(
        description='Creating heatmap')
    parser.add_argument('--umap_transform', required=False,
                        metavar="str", type=str,
                        help='folder for umap transform')
    parser.add_argument('--resolution', required=False,
                        metavar="int", type=int,
                        help='resolution of heatmap')
    parser.add_argument('--path', required=True,
                        metavar="str", type=str,
                        help='path to tiff files')
    parser.add_argument('--table', required=False,
                        metavar="str", type=str,
                        help='path to table file')
    parser.add_argument('--type', required=True,
                        metavar="str", type=str,
                        help='U2MAP or U3MAP')
    args = parser.parse_args()
    return args


def load_umap_transform(name):
    """
    Function to load the necessary models.
    A folder can contain two models (PCA+UMAP) or one (UMAP)
    Parameters
    ----------
    name: string, 
        folder name where to find the models.
    Returns
    -------
    A function to apply to a table, the function is:
    - the sequential application of PCA+UMAP
    - application of UMAP
    """
    files = os.listdir(name)
    if len(files) == 2:
        pca = pickle.load(open(os.path.join(name, files[0]), 'rb'))
        umap = pickle.load(open(os.path.join(name, files[1]), 'rb'))
        def predict_pca(z):
            z = pca.transform(z)
            pred = umap.transform(z)
            return pred
        return predict_pca
    else:
        umap = pickle.load(open(os.path.join(name, files[0]), 'rb'))
        def predict(z):
            try:
                pred = umap.transform(z)
            except:
                pred = umap.transform(z) # really weird if this works.
            return pred
        return predict


def f(slide, line, shape_slide_level):
    """
    Modified function from package useful_wsi.
    Instead of a taken a point, this function takes a table line.
    Parameters
    ----------
    slide : wsi object,
        openslide object from which we extract.
    line : dictionnary like object,
        this line has two options: Centroid_x and Centroid_y
        corresponding to a point_l at a given level dimension.
    level : int,
        level of the associated point.
    shape_slide_level : tuple of integer,
        corresponding to the size of the slide at level 'level'.
    Returns
    -------
    Returns the coordinates at a resolution level 
    of a given nuclei whose coordinates are at a level 0.
    """
    x_0, y_0 = (line["Centroid_x"], line["Centroid_y"])

    size_x_l = shape_slide_level[1]
    size_y_l = shape_slide_level[0]
    size_x_0 = float(slide.level_dimensions[0][0])
    size_y_0 = float(slide.level_dimensions[0][1])

    x_l = x_0 * size_x_l / size_x_0
    y_l = y_0 * size_y_l / size_y_0

    point_l = (round(x_l), round(y_l))

    return point_l


def from_cell_to_heatmap(slide, trans, cell_table, filter_out="LBP", level=7, n_comp=2):
    """
    
    Parameters
    ----------
    slide : wsi object,
        openslide object from which we extract.
    trans : function,
        infers the new coordinates of a given point. It is or:
            - the sequential application of PCA+UMAP
            - application of UMAP.
    cell_table : pandas dataframe,
        patient table, where each line corresponds to a nucleus.
    filter_out: str,
        String pattern to filter out columns from the feature table, in 'glob' form. 
        If pattern in the feature name, exclude feature.
    level : int,
        level of the resulting heatmap.
    n_comp : int,
        number of components after UMAP projection.
    Returns
    -------
    Returns a heatmap with the projected components of a given slide
    at a given resolution.
    """
    slide = open_image(slide)
    f1, f2 = normalise_csv(cell_table)
    feat = f1.columns
    feat = [el for el in feat if filter_out not in el]
    f1 = f1[feat]
    f1 = drop_na_axis(f1)
    standard_embedding = trans(f1)
    x = standard_embedding[:, 0]
    y = standard_embedding[:, 1]
    
    if level < slide.level_count:
        # if the pyramid scheme has a the png at the correct resolution
        shape_slide_level = get_whole_image(slide, level=level, numpy=True).shape
        within_slide_levels = True
    else:
        # if the pyramid scheme doesn't have the png at the correct resolution
        within_slide_levels = False
        high_pyramid_level = slide.level_count - 1 
        power = level - high_pyramid_level 
        shape_slide_level = get_whole_image(slide, level=high_pyramid_level, numpy=True).shape
        shape_slide_level = tuple((int(shape_slide_level[0] / (2 ** power)),  
                                  int(shape_slide_level[1] / (2 ** power)),
                                  3))
    xshape, yshape = shape_slide_level[0:2]
    f2["coord_l"] = f2.apply(lambda row: f(slide, row, shape_slide_level), axis=1)
    heatmap = np.zeros(shape=(xshape, yshape, 3))

    f1 = f1.reset_index(drop=True)
    f2 = f2.reset_index(drop=True)
    for coord_l, group in tqdm(f2.groupby("coord_l")):
        y_l, x_l = [int(el) for el in coord_l]
        heatmap[x_l, y_l, 0] = np.mean(x[group.index])
        heatmap[x_l, y_l, 1] = np.mean(y[group.index])
        if n_comp == 2:
            count = group.shape[0]
            heatmap[x_l, y_l, 2] = count
        else:
            z = standard_embedding[:, 2]
            heatmap[x_l, y_l, 2] = np.mean(z[group.index])
    return heatmap

def save_heat_map(name, arr):
    """
    Save heatmaps png.
    Parameters
    ----------
    name : string,
        name to save the numpy array to.
    arr : numpy array.
    """
    np.save(name, arr)

def main():
    options = options_parser()
    
    n_comp = int(options.umap_transform.split('MAP')[0][-1])
    level = options.resolution
    umap_transform = load_umap_transform(options.umap_transform)
    cell_table = options.table
    slide = os.path.join(options.path, os.path.basename(options.table))
    slide = slide.split('.cs')[0] + ".tiff"
    heat_map_3D = from_cell_to_heatmap(slide, umap_transform, cell_table, level=level, n_comp=n_comp)
    num = os.path.basename(cell_table).split('.')[0]
    num = os.path.basename(options.table).split('.')[0]
    name = "{}.npy".format(num)
    save_heat_map(name, heat_map_3D)


if __name__ == '__main__':
    main()
