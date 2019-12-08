nextflow run nextflow/UmapProjection.nf -resume -c ~/.nextflow/config -profile mines \
                                        --tiff_location /mnt/data3/pnaylor/Data/Combined_Biopsy \
                                        --table_location ./outputs/combined_data_1-0/cell_tables \
                                        --infer 0
