#!/usr/bin/env nextflow

CWD = System.getProperty("user.dir")

params.PROJECT_NAME = "TEST"
params.PROJECT_VERSION = "1-0"
output_folder = "./outputs/${params.PROJECT_NAME}_${params.PROJECT_VERSION}"

params.tiff_location = "./data/*.tiff" // tiff files to process
params.tissue_segmentation_model = "./meta/tissue_segmentation_model"
params.nucleus_segmentation_model = "./meta/nuclei_segmentation_model"
params.save = 1
TISSUE_SEG_MODEL = file(params.tissue_segmentation_model)
NUCLEUS_SEG_MODEL = file(params.nucleus_segmentation_model)
params.margin = 50
size = 1948
// input file
DATA = file(params.tiff_location)

// input scalar
wsi_margin = params.margin



process SegmentTissue {
    beforeScript "source ${CWD}/environment/GPU_LOCKS/set_gpu.sh ${CWD}"
    afterScript  "source ${CWD}/environment/GPU_LOCKS/free_gpu.sh ${CWD}"
    
    publishDir "${output_folder}/wsi_seg_checks/${s_without_ext}", mode: 'copy', overwrite: 'true', pattern: "*.png"

    input:
        file sample from DATA
        file model from TISSUE_SEG_MODEL
    output:
        set file(sample), file("${s_without_ext}_mask.png") into WSI_MASK
        file("${s_without_ext}__img.png")
        file("${s_without_ext}__overlay.png")
        file("${s_without_ext}_prob.png")
    script:
        s_without_ext = "${sample}".split("\\.")[0]
        template 'segmentation/tissue.py'
}


process TilePatient {

    beforeScript "source ${CWD}/environment/GPU_LOCKS/set_gpu.sh ${CWD}"
    afterScript  "source ${CWD}/environment/GPU_LOCKS/free_gpu.sh ${CWD}"

    publishDir "${output_folder}/wsi_seg_checks/${s_without_ext}", mode: 'copy', overwrite: 'true', pattern: "*.png"

    input:
        set file(sample), file(mask) from WSI_MASK
        file model from NUCLEUS_SEG_MODEL
    output:
        set file(sample), file("segmented_tiles.npz") into BATCH_SEG
        // add some pngs    
    script:
        s_without_ext = "${sample}".split("\\.")[0]
        template 'segmentation/tilling.py'
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
            coloring_tiles = file("./src/colouring/colouring.py")
            s_without_ext = "${sample}".split("\\.")[0]
            output_process = "${output_folder}/intermediate-tiles/${s_without_ext}/"
            """
            python $coloring_tiles --input $bag_segmentation --slide $sample
            """
    }

    process FeatureExtraction_CellMarking {
        publishDir "${output_process}", overwrite: true, pattern: "*.csv"
        publishDir "${output_process_tiles}", overwrite: true, pattern: "tiles_markedcells"

        memory { 4.GB  + 4.GB * task.attempt }
        errorStrategy 'retry'

        input:
            set file(sample), file(batch) from BATCH_VISU

        output:
            set file(sample), file("*.csv") into TABLE
            set file(sample), file("tiles_markedcells") into MARKED_CELLS

        script:
            output_process = "${output_folder}/cell_tables"
            s_without_ext = "${sample}".split("\\.")[0]
            output_process_tiles = "${output_folder}/intermediate-tiles/${s_without_ext}"
            marked_cells = "tiles_markedcells"
            extractor = file("src/extractor/extraction.py")
            """
            python $extractor --segmented_batch $batch \
                            --marge $wsi_margin \
                            --output_tiles $marked_cells \
                            --name $s_without_ext
            """
    }
    CONTOURS .concat(BIN, PROB, MARKED_CELLS).set{TO_STICH}

    process StichingTiff {
        publishDir "${output_process}", overwrite: true

        memory { 10.GB * task.attempt }
        errorStrategy 'retry'

        input:
        set file(sample), file(fold) from TO_STICH

        output:
        file "${wsi_sample}"

        script:
        writting_tiff = file("./src/stiching/create_wsi.py")
        s_without_ext = "${sample}".split("\\.")[0]
        wsi_sample = "${s_without_ext}_${fold.name.split('_')[1]}.tif"
        output_process = "${output_folder}/wsi/${s_without_ext}"
        """
        python $writting_tiff --input $fold \
                            --output $wsi_sample \
                            --slide $sample \
                            --marge $wsi_margin
        """
    }
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

        memory { 4.GB  + 4.GB * task.attempt }
        errorStrategy 'retry'

        input:
            set file(sample), file(batch) from BATCH_VISU

        output:
            set file(sample), file("*.csv") into TABLE

        script:
            output_process = "${output_folder}/cell_tables"
            s_without_ext = "${sample}".split("\\.")[0]
            output_process_tiles = "${output_folder}/intermediate-tiles/${s_without_ext}"
            marked_cells = "tiles_markedcells"
            extractor = file("src/extractor/extraction.py")
            """
            python $extractor --segmented_batch $batch \
                            --marge $wsi_margin \
                            --output_tiles $marked_cells \
                            --name $s_without_ext \
                            --no_samples
            """
    }

}