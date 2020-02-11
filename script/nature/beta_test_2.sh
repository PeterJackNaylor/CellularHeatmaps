nextflow run nextflow/Classification.nf -resume -c ~/.nextflow/config -profile mines \
                                        --PROJECT_NAME nature --PROJECT_VERSION 1-0 \
                                        --type U2MAP_reposition --resolution 7 \
                                        --label /mnt/data3/pnaylor/CellularHeatmaps/outputs/label_nature.csv
