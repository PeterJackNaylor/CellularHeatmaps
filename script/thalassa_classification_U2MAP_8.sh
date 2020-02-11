
label="/mnt/data3/pnaylor/CellularHeatmaps/outputs/label.csv"
nextflow run nextflow/Classification.nf -resume -c ~/.nextflow/config -profile mines \
                                        --PROJECT_NAME Combined_data --PROJECT_VERSION 1-0 \
                                        --label ${label} --resolution 8 --type U2MAP_reposition
