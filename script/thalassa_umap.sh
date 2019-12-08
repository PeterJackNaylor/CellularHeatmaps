nextflow run nextflow/UmapProjection.nf -resume -c ~/.nextflow/config -profile home \
                                        --tiff_location /mnt/data3/pnaylor/Data/Combined_Biopsy \
                                        --table_location ./outputs/combined_data/cell_tables \
                                        --infer 0
