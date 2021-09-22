

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import os
import pickle
import pandas as pd
from glob import glob
import numpy as np
# import dask.dataframe as dd
from tqdm import tqdm
import umap

from skimage.io import imsave
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from useful_wsi import get_image


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
    parser.add_argument('--path', required=False, default="*.csv",
                        metavar="str", type=str,
                        help='path')
    parser.add_argument('--n_components', required=True,
                        metavar="int", type=int,
                        help='2 or 3, 2 being umap to 2 + cell counts and 3 being umap to 3')
    parser.add_argument('--downsample_patient', required=False,
                        metavar="int", type=int,
                        help='downsampling rate for each patient individually')
    parser.add_argument('--downsample_whole', required=False,
                        metavar="int", type=int,
                        help='downsampling rate for the table as a whole after regroup everyone and how')
    parser.add_argument('--how', required=True, default="min",
                        metavar="str", type=str,
                        help='method to balance once you have everyone')
    parser.add_argument('--balance', required=False, default=0,
                        metavar="int", type=int,
                        help='if to balance each patient')
    parser.add_argument('--pca', required=False, default=0,
                        metavar="int", type=int,
                        help='if to use PCA or not')
    parser.add_argument('--plotting', required=False, default=0,
                        metavar="int", type=int,
                        help='if to plot..')
    args = parser.parse_args()
    args.balancing = args.balance == 1
    args.use_PCA = args.pca == 1
    args.plotting = args.plotting == 1
    if args.use_PCA:
        print('Using PCA')
    return args

def normalise_csv(f_csv, normalise=True, downsampling=1):
    """
    Loading function for the cell csv tables.
    Parameters
    ----------
    fcsv: string, 
        csv to load.
    normalise: bool,
        whether to normalise by the mean and standard deviation of the current csv table.
    downsampling: int,
        factor by which to downsample.
    Returns
    -------
    A tuple of tables, the first being the features and the second the information relative to the line.
    """
    xcel = pd.read_csv(f_csv, index_col=0)
    feat = xcel.columns[:-6]
    info = xcel.columns[-6:]
    xcel_feat = xcel.loc[:, feat]
    xcel_info = xcel.loc[:, info]
    if normalise:
        for el in feat:
            xcel_feat_mean = xcel_feat.loc[:, el].mean()
            xcel_feat_std = xcel_feat.loc[:, el].std()
            xcel_feat[el] = (xcel_feat.loc[:, el] - xcel_feat_mean) / xcel_feat_std
    if downsampling != 1:
        sample = np.random.choice(xcel_feat.index, size=xcel_feat.shape[0] // downsampling, replace=False)
        xcel_feat = xcel_feat.ix[sample]
        xcel_info = xcel_info.ix[sample]
    return xcel_feat, xcel_info

def collect_files(prefix="data/*.csv", downsampling=1):
    """
    Collect and opens files.
    Parameters
    ----------
    prefix: string,
        folder where to collect the cell csv tables.
    downsampling: int,
        factor by which to downsample.
    Returns
    -------
    dictionnary where each file name is associate to its cell table.
    """
    files = {}
    for f in tqdm(glob(prefix)):
        f_name = os.path.basename(f).split('.')[0]
        files[f_name] = normalise_csv(f, normalise=True, downsampling=downsampling)
    return files

def load_all(f, filter_out="LBP", balance=True, how="min", downsampling=1):
    """
    Loading function for the cell csv tables.
    Parameters
    ----------
    f: dictionnary, 
        where each key represents a tissue and each item a feature table.
    filter_out: str,
        String pattern to filter out columns from the feature table, in 'glob' form. 
        If pattern in the feature name, exclude feature.
    balance: bool,
        Whether to balance the number of cell per patients with 'how' method.
    how: str,
        The method for balancing:
            - min, look at the smallest amount of cell in one patient, use this as 
            number of sample to pick from each patients.
            - minthresh, same as before expect you cap the minimum in case this one
            would go to low..
    Returns
    -------
    A tuple of tables, the first being the concatenate features table of all patients.
    The second being the concatenated information relative to each cell of each patient.
    """
    tables = []
    tables_f2 = []
    for k in f.keys():
        f1, f2 = f[k]
        f2['patient'] = k
        feat = f1.columns
        feat = [el for el in feat if filter_out not in el]
        tables.append(f1[feat])
        tables_f2.append(f2)
    if balance:
        if how == "min":
            n_min = min([len(t) for t in tables])
            for i in range(len(tables)):
                sample = np.random.choice(tables[i].index, size=n_min, replace=False)
                tables[i] = tables[i].ix[sample]
                tables_f2[i] = tables_f2[i].ix[sample]
        elif how == "minthresh":
            min_thresh = 10000
            n_min = min([len(t) for t in tables])
            n_min = max(n_min, min_thresh)
            for i in range(len(tables)):
                size_table = tables[i].shape[0]
                sample = np.random.choice(tables[i].index, size=min(n_min, size_table), replace=False)
                tables[i] = tables[i].ix[sample]
                tables_f2[i] = tables_f2[i].ix[sample]
    print("Starting concatenation")
    output = pd.concat(tables, axis=0, ignore_index=True)
    output_info = pd.concat(tables_f2, axis=0, ignore_index=True)
    if downsampling != 1:
        sample = np.random.choice(output.index, size=output.shape[0] // downsampling, replace=False)
        output = output.ix[sample]
        output_info = output_info.ix[sample]
    return output, output_info

def drop_na_axis(table):
    """
    drop na from table.
    Parameters
    ----------
    table: pandas dataframe table, 
        we will drop all na from this table.
    Returns
    -------
    The table without the na values.
    """
    if table.isnull().values.any():
        before_feat = table.shape[1]
        table = table.dropna(axis=1)
        print("We dropped {} features because of NaN".format(before_feat - table.shape[1]))
    print("We have {} segmented cells and {} features fed to the umap".format(table.shape[0], table.shape[1]))
    return table

def normalise_again_f(table):
    """
    Normalising function for a table.
    Parameters
    ----------
    table: pandas dataframe table, 
        table to normalise by the mean and std.
    Returns
    -------
    Normalised table.
    """
    feat = table.columns
    for el in feat:
        table_mean = table.loc[:, el].mean()
        table_std = table.loc[:, el].std()
        table[el] = (table.loc[:, el] - table_mean) / table_std
    return table, table_mean, table_std

def filter_columns(table, filter_in=None, filter_out=None):
    """
    Table column filter function.
    After this function, the remaining feature will satisfy:
    being in filter_in and not being in filter_out.
    Parameters
    ----------
    table: pandas dataframe table, 
        table to normalise by the mean and std.
    filter_in: string,
        if the pattern is in the feature, keep the feature.
    filter_out: string,
        if the pattern is in the feature, exclude the feature.
    Returns
    -------
    Table with filtered columns.
    """
    feat = table.columns
    if filter_in is not None:
        feat = [el for el in feat if filter_in in feat]
    if filter_out is not None:
        feat = [el for el in feat if filter_out not in el]
    return table[feat]

def umap_plot(umap_transform, name, table, n_comp=2):
    """
    UMAP plot function, performs a density plot of the projected points.
    The points belong to the table and can be projected to two or three dimensions with 
    the umap transform function.
    Parameters
    ----------
    umap_transform: trained umap model.
    name: string,
        name of the saved plot.
    table: csv table,
        lines of table to project and plot. They have to be the exact same as those used 
        for training the umap.
    Returns
    -------
    A plot named 'name'.
    """
    fig = plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    x = drop_na_axis(table)
    standard_embedding = umap_transform(table)
    if n_comp == 2:
        x = standard_embedding[:, 0]
        y = standard_embedding[:, 1]
        data = pd.DataFrame({'x':x, 'y':y})
        sns.lmplot('x', 'y', data, fit_reg=False, scatter_kws={"s": 1.0})# , hue=)
    else:
        x = standard_embedding[:, 0]
        y = standard_embedding[:, 1]
        z = standard_embedding[:, 2]
        data = pd.DataFrame({'x':x, 'y':y, 'z':z})
        fig, axes = plt.subplots(ncols=3, figsize=(18, 16))
        sns.regplot('x', 'y', data, fit_reg=False, scatter_kws={"s": 1.0}, ax=axes[0])
        sns.regplot('x', 'z', data, fit_reg=False, scatter_kws={"s": 1.0}, ax=axes[1])
        sns.regplot('y', 'z', data, fit_reg=False, scatter_kws={"s": 1.0}, ax=axes[2])
    plt.savefig(name)

def pick_samples_from_cluster(data, info, y_pred, name, n_c=20, to_plot=10):
    """
    Picks samples from different clusters in a UMAP.
    Each point corresponds to an area in the wsi.
    Parameters
    ----------
    data: csv table,
        data that has been projected and plotted. 
    info: csv table
        same shape as data, each line corresponds to the information of the same line in data.
    y_pred: int vector,
        as long as the number of lines in data, each integer corresponds to a cluster.
    name: string,
        name of the resulting folder.
    n_c: int,
        number of clusters
    to_plot: int,
        how many samples to pick from each cluster.
    Returns
    -------
    A folder with with to_plot samples from each n_c clusters.
    """
    ind = 0
    for c in tqdm(range(n_c)):
        check_or_create(name + "/cluster_{}".format(c))
        index_table_y_pred_class = info.loc[y_pred == c].index
        sample = np.random.choice(index_table_y_pred_class, size=to_plot, replace=False)
        for _ in range(sample.shape[0]):
            patient_slide = os.path.join(data, info.ix[sample[_], 'patient']+".tiff")
            ind += 1
            img_m = get_cell(sample[_], info, slide=patient_slide, marge=20)
            crop_name = name + "/cluster_{}/{}_{}.png".format(c, os.path.basename(patient_slide).split('.')[0],ind)
            imsave(crop_name, img_m)

def get_cell(id_, table, slide, marge=0):
    """
    Returns the crop encapsulating the cell whose id is 'id_'
    Parameters
    ----------
    id_: int,
        corresponds to an element in the index of table 
    table: csv table
        table with all identified cell.
    slide: wsi,
        slide corresponding to the table
    marge: int,
        whether to take a margin when croping around the cell.
    Returns
    -------
    A crop arround nuclei id_.
    """
    info = table.loc[id_]
    x_min = int(info["BBox_x_min"])
    y_min = int(info["BBox_y_min"])
    size_x = int(info["BBox_x_max"] - x_min)
    size_y = int(info["BBox_y_max"] - y_min)
    para = [x_min - marge, y_min - marge, size_x + 2*marge, size_y+2*marge, 0]
    return get_image(slide, para)

def umap_plot_kmeans(umap_transform, name, table, info, 
                     n_comp=2, n_c=10, samples_per_cluster=50, 
                     path_to_slides="/mnt/data3/pnaylor/Data/Biopsy"):
    """
    UMAP plot function, performs a scatter plot of the projected points.
    In addition to ploting them, this function will give a colour to each point
    corresponding to its cluster assignement.
    Parameters
    ----------
    umap_transform: trained umap model.
    name: string,
        name of the saved plot.
    table: csv table,
        lines of table to project and plot. They have to be the exact same as those used 
        for training the umap.
    info: csv table,
        corresponding information table to table.
    n_comp: int, 
        number of connected components to project to.
    n_c: int,
        number of clusters
    samples_per_cluster: int,
        number of samples to pick from each cluster for visualisation purposes.
    Returns
    -------
    A plot named 'name' and a folder named with samples from each cluster.
    """
    check_or_create(name)
    fig = plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    x = drop_na_axis(table)
    standard_embedding = umap_transform(table)
    if n_comp == 2:
        x = standard_embedding[:, 0]
        y = standard_embedding[:, 1]
        y_pred = KMeans(n_clusters=n_c, random_state=42).fit_predict(np.array([x, y]).T)
        data = pd.DataFrame({'x':x, 'y':y, 'cluster':y_pred})
        sns.lmplot('x', 'y', data, hue="cluster", fit_reg=False, scatter_kws={"s": 1.0})
        pick_samples_from_cluster(path_to_slides, info, y_pred, name, n_c=n_c, to_plot=samples_per_cluster)
    else:
        x = standard_embedding[:, 0]
        y = standard_embedding[:, 1]
        z = standard_embedding[:, 2]
        y_pred = KMeans(n_clusters=n_c, random_state=42).fit_predict(np.array([x, y, z]).T)

        fig, axes = plt.subplots(ncols=3, figsize=(18, 16))
        axes[0].scatter(x, y, s=1., c=y_pred, cmap='tab{}'.format(n_c))
        axes[1].scatter(x, z, s=1., c=y_pred, cmap='tab{}'.format(n_c))
        img_bar = axes[2].scatter(y, z, s=1., c=y_pred, cmap='tab{}'.format(n_c))
        bounds = np.linspace(0,n_c,n_c+1)
        plt.colorbar(img_bar, ax=axes[2], spacing='proportional', ticks=bounds)
        pick_samples_from_cluster(path_to_slides, info, y_pred, name, n_c=n_c, to_plot=samples_per_cluster)
    plt.savefig(name + "/cluster_umap.pdf")


def umap_plot_patient(umap_transform, name, table, patient, n_comp=2):
    """
    UMAP plot function, performs a scatter plot or densty plot for a patient's tissue.
    Parameters
    ----------
    umap_transform: trained umap model.
    name: string,
        name of the saved plot.
    table: csv table,
        lines of table to project and plot. They have to be the exact same as those used 
        for training the umap.
    patient: string,
        patient id as found in table
    n_comp: int, 
        number of connected components to project to.
    Returns
    -------
    A plot named 'name' representing the scatter plot of the project points of
    a given patient.
    """
    fig = plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    x = drop_na_axis(table)
    standard_embedding = umap_transform(table)
    if n_comp == 2:
        x = standard_embedding[:, 0]
        y = standard_embedding[:, 1]
        data = pd.DataFrame({'x':x, 'y':y, 'patient':patient})
        sns.lmplot('x', 'y', data, hue="patient", fit_reg=False, scatter_kws={"s": 1.0})
    else:
        x = standard_embedding[:, 0]
        y = standard_embedding[:, 1]
        z = standard_embedding[:, 2]
        data = pd.DataFrame({'x':x, 'y':y, 'z':z, 'patient':patient})
        fig, axes = plt.subplots(ncols=3, figsize=(18, 16))
        axes[0].scatter(x, y, s=1., c=patient, cmap='Spectral')
        axes[1].scatter(x, z, s=1., c=patient, cmap='Spectral')
        img_bar = axes[2].scatter(y, z, s=1., c=patient, cmap='Spectral')
        plt.colorbar(img_bar, ax=axes[2], spacing='proportional')

    plt.savefig(name)




def train_umap(table, use_PCA=True, keep_axis=0.75, n_comp=2):
    """
    UMAP training function, performs a training of a umap with/or not a preprocessing step.
    Parameters
    ----------
    table: csv table,
        samples are lines of the table. The UMAP will be trained on them.
    use_PCA: bool,
        whether to use PCA or not, in practice, I wouldn't use it.
    keep_axis: float between 0 and 1,
        percentage of axis to keep if using PCA.
    n_comp: int, 
        number of connected components to project to.
    Returns
    -------
    A tuple where the first element corresponds to a predict function
    and the second to the list of models necessary for predicting. (only UMAP or PCA+UMAP)
    """
    x = drop_na_axis(table)
    feat = x.columns
    if use_PCA:
        print('Using PCA')
        n = int(len(feat) * keep_axis)
        print("Keeping {} axis for PCA.".format(n))
        pca = PCA(n_components=n)
        x = pca.fit_transform(x[feat]) 
    else:
        x = x[feat]
    print("Starting training")
    trans = umap.UMAP(n_components=n_comp, random_state=42, 
                      n_neighbors=10, min_dist=0.).fit(x)
    print("Training over")
    if use_PCA:
        objects = [pca, trans]
        def predict_pca(z):
            z = pca.transform(z)
            pred = trans.transform(z)
            return pred
        return predict_pca, objects
    else:
        objects = [trans]
        def predict(z):
            pred = trans.transform(z)
            return pred
        return predict, objects

def save_umap(name, objects):
    """
    UMAP saving function, saves a list of prediction models
    in folder named 'name' by pickling each model in objects.
    Parameters
    ----------
    name: string,
        name of the folder to save to.
    objects: list of models,
        List of models to save.
    """
    check_or_create(name)
    n_umap = os.path.join(name, "umap.pkl")
    if len(objects) == 2:
        n_pca = os.path.join(name, "pca.pkl")
        pca, umap = objects
        pickle.dump(pca, open(n_pca, 'wb'))
    else:
        umap = objects[0]
    pickle.dump(umap, open(n_umap, 'wb'))


def main():
    options = options_parser()
    files = collect_files(options.path, downsampling=options.downsample_patient)
    table_cells, table_info = load_all(files, 
                                       balance=options.balancing, 
                                       how=options.how, 
                                       downsampling=options.downsample_whole)
    print("Columns: ", table_cells.columns)
    print("Shape of table cells :", table_cells.shape)
    normalise_again = False
    if normalise_again:
        table_cells, mean, std = normalise_again_f(table_cells)
    table_cells = filter_columns(table_cells, filter_in=None, filter_out="LBP")
    umap_transform, objects = train_umap(table_cells, options.use_PCA, 
                                         keep_axis=0.75, 
                                         n_comp=options.n_components)
    
    name_object = "model_U{}MAP".format(options.n_components)
    save_umap(name_object, objects)

    if options.plotting:
        name_file = "umap_cell_plot_ncomp_{}.pdf".format(options.n_components)
        # name_file_pat = "umap_cellpatient_plot_ncomp_{}.pdf".format(options.n_components)
        name_file_kmeans = "umap_cellkmeans_ncomp_{}".format(options.n_components)

        umap_plot(umap_transform, name_file, table_cells, 
                  n_comp=options.n_components)

        # umap_plot_patient(umap_transform, name_file_pat, table_cells, 
        #                   table_info['patient'], n_comp=options.n_components)
        umap_plot_kmeans(umap_transform, name_file_kmeans, 
                         table_cells, table_info, n_comp=options.n_components, n_c=10)


if __name__ == '__main__':
    main()