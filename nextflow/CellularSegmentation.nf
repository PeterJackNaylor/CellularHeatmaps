#!/usr/bin/env nextflow

params.PROJECT_NAME = "TEST"
params.PROJECT_VERSION = "1-0"
output_folder = "./outputs/${params.PROJECT_NAME}_${params.PROJECT_VERSION}"

params.tiff_location = "../Data/Biopsy" // tiff files to process
params.tissue_bound_annot = "../Data/Biopsy/tissue_segmentation" // xml folder containing tissue segmentation mask for each patient
params.segmentation_weights = "../test_judith_project/tmp/test_tcga_project/model/Distance111008_32" // segmentation weights
params.segmentation_mean = "../test_judith_project/tmp/test_tcga_project/model/mean_file_111008.npy" // segmentation mean file
params.margin = 50

// input file
tiff_files = file(params.tiff_location + "/*.tiff")
boundaries_files = file(params.tissue_bound_annot)


// input scalar
wsi_margin = params.margin


process TilePatient {
    publishDir "${output_folder}/tiling/", pattern: "*.png", overwrite: true
    
    input:
    file x from tiff_files

    output:
    set  val("$name"), file(x), file("${name}_tiles.txt") into patient_tile_info_i
    file("${name}_visualisation.png")

    script:
    tiling_py = file("./python/tiling/tiling_info.py")
    name = x.baseName
    xml_file = file(boundaries_files + "/${name}.xml")
    """
    export CONDA_PATH_BACKUP=""
    export PS1=""
    source activate cpu_env
    python $tiling_py \
                 --slide $x \
                 --output $name \
                 --marge $wsi_margin \
                 --tissue_seg $xml_file
    """
}

process TileSegmentation {
    publishDir "${output_process}", overwrite: true

    queue "gpu-cbio"
    memory '10GB'
    clusterOptions "--gres=gpu:1 --exclude=node[28]"
    // scratch true
    maxForks 16

    input:
    set name, file(slide), file(info) from patient_tile_info_i

    output:
    set name, file("tiles_h5") into bag_seg_i

    script:
    gpu_segmentation = file("./python/nuclei_segmentation/gpu_segmentation.py")
    seg_weights = file(params.segmentation_weights)
    seg_mean = file(params.segmentation_mean)
    output = "./tiles_h5/"
    output_process = "${output_folder}/intermediate-tiles/${name}"
    """
    module load cuda10.0 
    python $gpu_segmentation \
                 --slide $slide \
                 --parameter $info \
                 --output $output \
                 --log $seg_weights \
                 --mean $seg_mean
    """
}

process Colouring {
    publishDir "${output_process}", overwrite: true

    input:
    set name, file(bag_segmentation) from bag_seg_i

    output:
    set name, file("tiles_rgb"), file("tiles_bin") into rgb_and_bin
    set name, file("tiles_bin") into bin
    set name, file("tiles_contours") into contours
    set name, file("tiles_prob") into dist 

    script:
    coloring_tiles = file("./python/colouring/colouring.py")
    output_process = "${output_folder}/intermediate-tiles/${name}/"
    """
    python $coloring_tiles --input '$bag_segmentation/*.h5'
    """
}

process FeatureExtraction {
    publishDir "${output_process}", overwrite: true, pattern: "*.csv"
    publishDir "${output_process_tiles}", overwrite: true, pattern: "tiles_markedcells"

    memory { 4.GB  + 4.GB * task.attempt }
    errorStrategy 'retry'

    input:
    set name, file(rgb), file(bin) from rgb_and_bin

    output:
    set name, file("*.csv") into table
    set name, file("tiles_markedcells") into marked_cells

    script:
    output_process = "${output_folder}/cell_tables"
    output_process_tiles = "${output_folder}/intermediate-tiles/${name}"
    marked_cells = "tiles_markedcells"
    extractor = file("python/extractor/extraction.py")
    """
    python $extractor --rgb_folder $rgb \
                      --bin_folder $bin \
                      --marge $wsi_margin \
                      --output_tiles $marked_cells \
                      --name $name
    """
}

contours .concat(bin, dist, marked_cells).set{for_stiching}

process StichingTiff {
    publishDir "${output_process}", overwrite: true

    memory { 10.GB * task.attempt }
    errorStrategy 'retry'

    input:
    set name, file(fold) from for_stiching

    output:
    file "${wsi_name}"

    script:
    writting_tiff = file("./python/stiching/create_wsi.py")
    wsi_name = "${name}_${fold.name.split('_')[1]}.tif"
    slide = file(params.tiff_location + "/${name}.tiff")
    output_process = "${output_folder}/wsi/${name}"
    """
    python $writting_tiff --input $fold \
                          --output $wsi_name \
                          --slide $slide \
                          --marge $wsi_margin
    """
}
