
# CellularHeatmaps

CellularHeatmaps github repository contains all the necessary code to reproduce the models and analysis in [CITE PAPER].

# Pipeline -- Nextflow

## On thalassa
1) Generate labels with the given folder

'bash script/thalassa_fab_tnbc.sh'

'bash script/thalassa_substrat.sh'

2) 


'bash script/thalassa_label.sh 

bash script/thalassa_umap.sh 

bash script/thalassa_classification_U2MAP_8.sh 
'

### Cellular Segmentation
The first cellular segmentation model

## Project nature: TNBC (Fabien + Substrat)

After generating the labels:
1) 0_data_prep.sh
2) 1_umap_generation.sh
3) 2_classification.sh




## TODO
- fix name out of repositioning, there is a trailing r in the name.
- documentation
- test
- requirements
- filter all patients
- extra files are annoying
- check it runs
- evaluation
- plots
