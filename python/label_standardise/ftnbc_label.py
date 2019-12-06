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


FEATURES = ['Biopsy',
            'Index mitotique',
            'Grade EE',
            '% stroma',
            '% cellules tumorales (dont CIS)',
            'cis/carcinome total',
            '%TIL',
            '% Stromal lymphocytes',
            'RCB',
            'RCB class'
            ]

def load_custom_xlsx(path):
    sheet_name = 'TRIPLE NEGATIF'
    df_features = pd.read_excel(path, sheetname=sheet_name)[FEATURES]

    df_features["Grade EE 1"] = (df_features['Grade EE'] == 1).astype(int)
    df_features["Grade EE 2"] = (df_features['Grade EE'] == 2).astype(int)
    df_features["Grade EE 3"] = (df_features['Grade EE'] == 3).astype(int)
    df_features = df_features.drop("Grade EE", axis=1)
    df_features = df_features.dropna()
    # remove patient with string in numeric column
    df_features = df_features.drop(32).reset_index()
    df_features['%TIL'] = df_features['%TIL'].astype('int')
    df_features = df_features.drop('index', axis=1)
    df_features = df_features.set_index('Biopsy')
    return df_features

def split(stri):
    return int(os.path.basename(stri).split('.')[0])

def f(val):
    if val == "pCR":
        res = 0
    elif val == "RCB-I":
        res = 1
    elif val == "RCB-II":
        res = 2
    else:
        res = 3
    return res

def main():
    options = get_options()
    table = load_custom_xlsx(options.input_table)
    which_tiff_exist = [split(f) for f in glob(options.folder_to_check + '/*.tiff')]
    table.columns = ["index_mitotique", "stroma", "cancer", "cis", "til", "stroma_lym",  'RCB', 'RCB_class', "grade_1", "grade_2", "grade_3"]
    table = table[["index_mitotique", "stroma", "cancer", "cis", "til", "stroma_lym", "grade_1", "grade_2", "grade_3", 'RCB', 'RCB_class']]
    table['RCB_class'] = table[["RCB_class"]].apply(lambda row: f(row["RCB_class"]), axis=1)

    table.ix[which_tiff_exist].to_csv(options.output_table)

if __name__ == '__main__':
    main()

