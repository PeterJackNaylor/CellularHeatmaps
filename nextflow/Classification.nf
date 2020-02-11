


params.PROJECT_NAME = "TEST"
params.PROJECT_VERSION = "1-0"
output_folder = "./outputs/${params.PROJECT_NAME}_${params.PROJECT_VERSION}"

params.type = "U2MAP"
params.resolution = "7"

params.input_fold = "${output_folder}/umap/patient_projection/${params.type}/${params.resolution}"
input_fold = file(params.input_fold)

params.label = "/mnt/data3/pnaylor/CellularHeatmaps/outputs/label.csv"
label = file(params.label)

output_folder = "${output_folder}/${params.type}_${params.resolution}/"

inner_fold =  5
batch_size = 16
epochs =  80
models = ['resnet50']
interests = ['Residual', 'Prognostic']
learning_rate = [10e-5, 10e-6, 10e-4]
dropout = 0.5
repeat = 10

process TrainValidTestPred {
    publishDir "${output_process}", overwrite: true

    queue "gpu-cbio"
    memory '30GB'
    clusterOptions "--gres=gpu:1"
//    scratch true

    input:
    file path from input_fold

    each lr from learning_rate
    each model from models  
    each y_interest from interests
    each n_test from 0..9

    output:
    set y_interest, file(score_names) into score_probability

    script:
    base_name = "${y_interest}_fold_${n_test}_model_${model}_lr_${lr}"
    output_process = "${output_folder}/classification/train"
    py_model = file("./python/classification/multiple_run.py")
    weight_names = base_name + ".h5"
    score_names = base_name + ".pkl"
    """
    module load cuda10.0 
    python $py_model --batch_size $batch_size \\
                     --path  $path \\
                     --labels $label \\
                     --y_interest $y_interest \\
                     --out_weight  $weight_names \\
                     --model $model \\
                     --epochs $epochs \\
                     --dropout $dropout \\
                     --repeat $repeat \\
                     --inner_fold $inner_fold \\
                     --multiprocess 0 \\
                     --workers 10 \\
                     --optimizer adam \\
                     --fold_test $n_test \\
                     --lr $lr \\
                     --filename $score_names 
    """
}

score_probability .groupTuple()
                  .set{score_probability_y_interest}

process pickle_collector {
    publishDir "${output_process}", overwrite: true

    memory '30GB'

    input:
    set y_interest, file(pickle) from score_probability_y_interest

    output:
    file "${filename}"

    script:
    py_model = file("./python/classification/aggregating_results.py")
    filename = "classification_${y_interest}_${params.type}_${params.resolution}.csv"
    output_process = "${output_folder}/classification/results"

    """
    python $py_model --labels $label \\
                     --inner_fold $inner_fold \\
                     --y_interest $y_interest \\
                     --filename $filename
    """
}