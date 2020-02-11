nextflow run nextflow/CellularSegmentation.nf -resume -c ~/.nextflow/config -profile mines \
                                           --PROJECT_NAME nature --PROJECT_VERSION 1-0 \
                                           --tiff_location ../Data/nature_medecine_biop/ \
                                           --tissue_bound_annot ../Data/nature_medecine_biop/tissue_segmentation \
                                           --segmentation_weights ../test_judith_project/tmp/test_tcga_project/model/Distance111008_32 \
                                           --segmentation_mean ../test_judith_project/tmp/test_tcga_project/model/mean_file_111008.npy