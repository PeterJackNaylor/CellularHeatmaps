import os

import pandas as pd
from glob import glob

def get_options():
    import argparse
    parser = argparse.ArgumentParser(
        description='Creating training on heatmaps')

    parser.add_argument('--input_table',
                        metavar="str", type=str,
                        help='original xml table')

    parser.add_argument('--folder_to_check',
                        metavar="str", type=str,
                        help='folder to compare with')

    parser.add_argument('--output_table', 
                        metavar="int", type=str,
                        help='output csv table name')

    args = parser.parse_args()
    return args


def f(val):
    if val == 0:
        res = 0
    elif val < 1.3:
        res = 1
    elif val < 3.28:
        res = 2
    else:
        res = 3
    return res

def load_custom_xlsx(path):
    df_features = pd.read_csv(path)

    df_features["grade_1"] = (df_features['Grade'] == 1).astype(int)
    df_features["grade_2"] = (df_features['Grade'] == 2).astype(int)
    df_features["grade_3"] = (df_features['Grade'] == 3).astype(int)

    df_features = df_features.drop("Grade", axis=1)
    df_features = df_features.ix[df_features[["rcb"]].dropna(axis=0).index]
    df_features["RCB_class"] = df_features[["rcb"]].apply(lambda row: f(row["rcb"]), axis=1)
    df_features = df_features.set_index('Biopsy')
    df_features = df_features[["Index mitotique", "TILS", "grade_1", "grade_2", "grade_3", "her2", "Ki67", "Necrose", "rcb", "RCB_class"]]
    df_features.columns = ["index_mitotique", "til", "grade_1", "grade_2", "grade_3", "her2", "ki67", "necrose", "RCB", "RCB_class"]
    return df_features

def split(stri):
    return os.path.basename(stri).split('.')[0]


def main():
    options = get_options()
    table = load_custom_xlsx(options.input_table)
    which_tiff_exist = [split(f) for f in glob(options.folder_to_check + '/*.tiff')]
    table.ix[which_tiff_exist].to_csv(options.output_table)

if __name__ == '__main__':
    main()

