#!/usr/bin/env nextflow

params.PROJECT_NAME = "Combined_data"
params.PROJECT_VERSION = "1-0"


output_folder = "./outputs/${params.PROJECT_NAME}_${params.PROJECT_VERSION}"

params.tiff_location = "../Data/Biopsy" // tiff files to process
wsi_folder = file(params.tiff_location)

params.table_location = "./output/${params.PROJECT_NAME}_${params.PROJECT_VERSION}/cell_tables"
tables = file(params.table_location + "/*.csv")

// parameters for the types.
components = [2, 3]
resolutions = [7, 8, 9]


if (params.infer == 1){
    // You can:
    // or specify a directory with umap trained (only one model)
    // or specify a project type and two umap model (2/3 component)
    params.model = "${output_folder}/umap/model/model_U{2,3}MAP"
    umap_models = Channel.from(file(params.model, type: "dir"))
    umap_models.map{el -> [el.name.split('_')[1], el]}.set{umap_model}
}else{
    process UmapTraining {
        publishDir "${output_process}", overwrite: true

        memory '120GB'

        input:
        file _ from Channel.from(tables).collect()
        each n_comp from components

        output:
        set val("U${n_comp}MAP"), file("model_U${n_comp}MAP") into umap_model

        script:
        umap_cell = file("./python/umap/train_umap.py")
        output_process = "${output_folder}/umap/model"
        """
        python $umap_cell --path '*.csv' \
                        --n_component $n_comp \
                        --downsample_patient 1 \
                        --downsample_whole 1 \
                        --how minthresh \
                        --balance 1 \
                        --pca 0 \
                        --plotting 0
        """
    }
}

process HeatUMAPGeneration {
    publishDir "${output_process}", overwrite: true

    memory { 15.GB * (1 + task.attempt) }
    errorStrategy 'retry'

    input:
    file table from tables
    each transf from umap_model
    each resolution from resolutions

    output:
    set val("$name"), val("$resolution"), val("$n_comp"), file("${name}.npy") into patient_heatumap

    script:
    n_comp = transf[0]
    model = file(transf[1])
    name = "${table.baseName}"
    patient_projection = file("python/umap/umap_infer.py")
    output_process = "${output_folder}/umap/patient_projection/${n_comp}/${resolution}"
    """
    python $patient_projection --resolution $resolution \
                               --umap_transform $model \
                               --path $wsi_folder \
                               --table $table \
                               --type $n_comp
    """
}

process Repositioning {
    publishDir "${output_process}", overwrite: true

    memory '15GB'

    input:
    set name, resolution, n_comp, file(heatmap) from patient_heatumap // groupby or something

    output:
    file("*.npy") into patient_prepro

    script:
    output_process = "${output_folder}/umap/patient_projection/${n_comp}_reposition/${resolution}"
    reposition = file("./python/repositioning/repositioning.py")
    """
    python $reposition --input $heatmap --output . 
    """
}