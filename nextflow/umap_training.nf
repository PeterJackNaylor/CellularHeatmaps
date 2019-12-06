#!/usr/bin/env nextflow

params.PROJECT_NAME = "TEST"
params.PROJECT_VERSION = "1-0"

if (params.infer == 1){
    params.model = "output/umap_training/"
    model = file(params.model)
}
output_folder = "./outputs/${params.PROJECT_NAME}_${params.PROJECT_VERSION}"

params.tiff_location = "../Data/Biopsy" // tiff files to process
params.table_location = "./output/${params.PROJECT_NAME}_${params.PROJECT_VERSION}"

tables = file(params.table_location + "/*.csv")
components = [2, 3]

if (params.infer == 1){
    umap_cell = file()
}else{
    process UmapTraining {
        publishDir "${output_process}", overwrite: true

        // memory '120GB'

        input:
        file _ from tables.collect()
        each n_comp from components

        output:
        file("umap_cell_transform_ncomp_*") into umap_cell_transform

        script:
        umap_cell = file("./python/umap/umap_cell_all.py")
        output_process = "${output_folder}/umap/"
        """
        python $umap_cell --path '*.csv' \
                        --n_component $n_comp \
                        --downsample_patient 1 \
                        --downsample_whole 1 \
                        --how minthresh \
                        --balance 1 \
                        --plotting 0
        """
    }
}



// umap_cell_patient = file("cell_heat_maps/umap_cell_project_patient.py")
// resolution = 8

// process CreateHeatUMap {
//     publishDir "outputs/UmapCell/patients/", overwrite: true, pattern: "*.npy"
//     tag { "UmapCell"}
//     memory '10GB'
//     input:
//     file table from table_umap2
//     each transf from umap_cell_transform
//     output:
//     set val("${table[0].baseName}"), file("heatmap_*") into patient_heatumap

//     """
//     python $umap_cell_patient --resolution $resolution \
//     --umap_transform $transf \
//     --path $data_folder \
//     --table ${table[1]} 
//     """
// }



// heat_map_creation = file("heatmap/heatmap_creation.py")


// process CreateHeatMap {
//     publishDir "outputs/heat_maps/${t.baseName.split('_')[0]}", overwrite: true, pattern: "*.npy"
//     tag { slide.baseName + " heatmap " }
//     memory '15GB'

//     input:
//     set file(slide), file(t) from table
//     output:
//     set val("${slide.baseName}"), file("*.npy") into patient_heatmaps

//     """
//     python $heat_map_creation --input_slide $slide --input_table $t --resolution $resolution
//     """
// }