import numpy as np
from glob import glob 
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os
def get_options():
    import argparse
    parser = argparse.ArgumentParser(
        description='Creating training on heatmaps')

    parser.add_argument('--substra',
                        metavar="str", type=str,
                        help='substra csv')

    parser.add_argument("--folder", type=str,
                        metavar="str", 
                        help='tiff folder for checking')

    parser.add_argument('--ftnbc',
                        metavar="str", type=str,
                        help='ftnbc csv')

    parser.add_argument('--output_name', 
                        metavar="int", type=str,
                        help='output csv table name')

    args = parser.parse_args()
    return args

def createfolds(table, num, strat_var):

    skf = StratifiedKFold(n_splits=num, shuffle=True)
    obj = skf.split(table.index, table[strat_var])
    i = 0
    for _, test_index in obj:
        table.ix[test_index, "fold"] = i
        i += 1
    return table

def main():

    options = get_options()
    substra = pd.read_csv(options.substra)
    ftnbc = pd.read_csv(options.ftnbc)
    ftnbc['Biopsy'] = ftnbc['Biopsy'].astype(str)
    mer = pd.concat([substra, ftnbc])
    mer["Residual"] = (mer["RCB"] == 0).astype('int')
    mer["Prognostic"] = (mer["RCB"] < 1.1).astype('int')
    mer = mer.reset_index(drop=True)
    mer = mer.ix[mer[["RCB"]].dropna().index]
    mer = mer.reset_index(drop=True)
    mer = mer.set_index('Biopsy')
    mer.index = mer.index.map(str)
    existing_files = glob(options.folder + '/*.tiff')
    try:
        name_out = os.path.basename(options.output_name)
        os.mkdir(options.output_name.replace(name_out, "") + '/not_in_table')
    except:
        pass
    names_to_keep = []
    for f in existing_files:
        name = os.path.basename(f)
        name = name.split('.')[0]
        name_out = os.path.basename(options.output_name)
        if name not in mer.index:
            options.output_name.replace(name_out, "") + '/not_in_table/' + str(name) +".tiff"
            os.rename(f, options.output_name.replace(name_out, "") + '/not_in_table/' + str(name) +".tiff" )
        else:
            names_to_keep.append(name)

    mer = mer.ix[np.array(names_to_keep)]
    mer = createfolds(mer, 10, 'RCB_class')
    mer.to_csv(options.output_name)







if __name__ == '__main__':
    main()

