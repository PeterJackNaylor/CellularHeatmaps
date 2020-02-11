
import pandas as pd 
from glob import glob

folder = '/Users/naylorpeter/tmptmp/nature/test_jpg'
csv_file = '/Users/naylorpeter/tmptmp/nature/label_nature_jpeg.csv'

table = pd.read_csv(csv_file)
table['Biopsy'] = table['Biopsy'].astype(str)
table = table.set_index('Biopsy')
files = glob(folder + "/*.jpg")

def tronc(n):
    return n.split('/')[-1].split('.')[0]

files = [(f, tronc(f)) for f in files]
for path, idx in files:
    table.ix[idx, "path"] = path

table = table[~(table.RCB.isna())]
table = table[~(table.path.isna())]

table.to_csv('test_jpg.csv')