nextflow run nextflow/UmapProjection.nf -resume -c ~/.nextflow/config -profile home \
                                        --tiff_location ../tiff \
                                        --table_location ./outputs/TEST_1-0/cell_tables \
                                        --infer 1
