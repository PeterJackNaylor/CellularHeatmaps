nextflow run nextflow/umap_training.nf -resume -c ~/.nextflow/config -profile home \
                                          --tiff_location ../tiff \
                                          --table_location ./output/TEST_1-0 \
                                           --infer 0
