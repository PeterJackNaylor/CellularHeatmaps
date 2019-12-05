nextflow run nextflow/CellularSegmentation.nf -resume -c ~/.nextflow/config -profile home \
                                          --tiff_location ../tiff \
                                           --tissue_bound_annot ../tissue_seg \
                                           --segmentation_weights ../segmentation_model/Distance111008_32 \
                                           --segmentation_mean ../segmentation_model/mean_file_111008.npy