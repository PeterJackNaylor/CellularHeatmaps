#!/usr/bin/env nextflow

CWD = System.getProperty("user.dir")

params.PROJECT_NAME = "TESTBACH"
params.PROJECT_VERSION = "1-0"
output_folder = "./outputs/${params.PROJECT_NAME}_${params.PROJECT_VERSION}"

params.tiff_location = "./data/*/" // tiff files to process
params.nucleus_segmentation_model = "./meta/nuclei_segmentation_model"
params.save = 1
params.n_jobs = 4
n_jobs = params.n_jobs
NUCLEUS_SEG_MODEL = file(params.nucleus_segmentation_model)
size_x = 1536
size_y = 2048
// input file
DATA = Channel.fromPath(params.tiff_location, type: 'dir')



process TilePatient {

    beforeScript "source ${CWD}/environment/GPU_LOCKS/set_gpu.sh ${CWD}"
    afterScript  "source ${CWD}/environment/GPU_LOCKS/free_gpu.sh ${CWD}"

    publishDir "${output_folder}/wsi_seg_checks/${s_without_ext}", mode: 'copy', overwrite: 'true', pattern: "*.png"

    input:
        file sample from DATA
        file model from NUCLEUS_SEG_MODEL
    output:
        set file(sample), file("segmented_tiles.npz") into BATCH_SEG
        // add some pngs    
    script:
        s_without_ext = "${sample}".split("\\.")[0]
        template 'segmentation/tilling_bach.py'
}

if (params.save == 1){
    process Colouring {
        publishDir "${output_process}", overwrite: true

        input:
        set file(sample), file(bag_segmentation) from BATCH_SEG

        output:
            set file(sample), file("tiles_bin") into BIN
            set file(sample), file("tiles_contours") into CONTOURS
            set file(sample), file("tiles_prob") into PROB
            set file(sample), file("segmented_tiles_and_bins.npz") into BATCH_VISU

        script:
            coloring_tiles = file("./src/colouring/colouring_bach.py")
            s_without_ext = "${sample}".split("\\.")[0]
            output_process = "${output_folder}/intermediate-tiles/${s_without_ext}/"
            """
            python $coloring_tiles --input $bag_segmentation --slide $sample
            """
    }

    process FeatureExtraction_CellMarking {
        publishDir "${output_process}", overwrite: true, pattern: "*.csv"
        publishDir "${output_process_tiny_cells}", overwrite: true, pattern: "*.npy"
        publishDir "${output_process_tiles}", overwrite: true, pattern: "tiles_markedcells"

        memory { 4.GB  + 4.GB * task.attempt }
        errorStrategy 'retry'

        input:
            set file(sample), file(batch) from BATCH_VISU

        output:
            set file(sample), file("*.csv") into TABLE
            set file(sample), file("tiles_markedcells") into MARKED_CELLS
            file("*.npy") into TINY_CELLS

        script:
            output_process = "${output_folder}/cell_tables"
            s_without_ext = "${sample}".split("\\.")[0]
            output_process_tiles = "${output_folder}/intermediate-tiles/${s_without_ext}"
            output_process_tiny_cells = "${output_folder}/tiny_cells"
            marked_cells = "tiles_markedcells"
            extractor = file("src/extractor/extraction.py")
            """
            python $extractor --segmented_batch $batch \
                            --marge 0 \
                            --output_tiles $marked_cells \
                            --name $s_without_ext --n_jobs $n_jobs
            """
    }
    CONTOURS .concat(BIN, PROB, MARKED_CELLS).set{TO_STICH}

} else {

    process Postprocessing {
        input:
        set file(sample), file(bag_segmentation) from BATCH_SEG

        output:
            set file(sample), file("segmented_tiles_and_bins.npz") into BATCH_VISU

        script:
            coloring_tiles = file("./src/colouring/colouring.py")
            s_without_ext = "${sample}".split("\\.")[0]
            """
            python $coloring_tiles --input $bag_segmentation --slide $sample --no_samples
            """
    }

    process FeatureExtraction {
        publishDir "${output_process}", overwrite: true, pattern: "*.csv"
        publishDir "${output_process_tiny_cells}", overwrite: true, pattern: "*.npy"

        memory { 4.GB  + 4.GB * task.attempt }
        errorStrategy 'retry'

        input:
            set file(sample), file(batch) from BATCH_VISU

        output:
            set file(sample), file("*.csv") into TABLE
            file("*.npy") into TINY_CELLS

        script:
            output_process = "${output_folder}/cell_tables"
            s_without_ext = "${sample}".split("\\.")[0]
            output_process_tiles = "${output_folder}/intermediate-tiles/${s_without_ext}"
            output_process_tiny_cells = "${output_folder}/tiny_cells"
            marked_cells = "tiles_markedcells"
            extractor = file("src/extractor/extraction.py")
            """
            python $extractor --segmented_batch $batch \
                            --marge 0 \
                            --output_tiles $marked_cells \
                            --name $s_without_ext \
                            --no_samples --n_jobs $n_jobs
            """
    }

}