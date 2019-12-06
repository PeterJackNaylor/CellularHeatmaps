

import pandas as pd
from sklearn.model_selection import StratifiedKFold

def get_options():
    import argparse
    parser = argparse.ArgumentParser(
        description='Creating training on heatmaps')

    parser.add_argument('--substra',
                        metavar="str", type=str,
                        help='substra csv')

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
    mer = pd.concat([substra, ftnbc])
    mer["Residual"] = (mer["RCB"] == 0).astype('int')
    mer["Prognostic"] = (mer["RCB"] < 1.1).astype('int')
    import pdb; pdb.set_trace()
    mer = createfolds(mer, 10, 'RCB_class')
    mer.set_index('Biopsy')
    mer.to_csv(options.output_table)

if __name__ == '__main__':
    main()

