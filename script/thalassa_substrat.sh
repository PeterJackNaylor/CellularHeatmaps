nextflow run nextflow/CellularSegmentation.nf -resume -c ~/.nextflow/config -profile mines \
                                          --tiff_location ../Data/Biopsy_guillaume \
                                           --tissue_bound_annot ../Data/Biopsy_guillaume/tissue_segmentation \
                                           --segmentation_weights ../test_judith_project/tmp/test_tcga_project/model/Distance111008_32 \
                                           --segmentation_mean ../test_judith_project/tmp/test_tcga_project/model/mean_file_111008.npy