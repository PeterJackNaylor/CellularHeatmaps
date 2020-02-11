nextflow run nextflow/UmapProjection.nf -resume -c ~/.nextflow/config -profile mines \
                                        --PROJECT_NAME nature --PROJECT_VERSION 1-0 \
                                        --tiff_location ../Data/nature_medecine_biop/ \
                                        --table_location ./outputs/nature_1-0/cell_tables \
                                        --infer 0
