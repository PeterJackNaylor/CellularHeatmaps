
training = file("training_test.py")

epochs =  80
inner_fold =  10
batch_size = 16
resolution = "7"
base = "/mnt/data3/pnaylor/ProjectFabien/outputs/UmapCell/resolution" + resolution + "/"
paths = [file(base + "repos_comp2"),  file(base + "repos_comp3"), file(base + "comp2"), , file(base + "comp3")]
models = ['resnet50convflat']
interests = ['RCB_class', 'RCB_classcan']
labels = file("/mnt/data3/pnaylor/ProjectFabien/outputs/multi_class.csv")
learning_rate = [10e-5, 10e-6, 10e-4]
dropout = 0.5
tvt = file("trainvalidtest.py")

process TrainValidTestPred {
    queue "gpu-cbio"
    memory '30GB'
    clusterOptions "--gres=gpu:1"
    scratch true
    input:
    each file(path) from paths
    each lr from learning_rate
    each model from models  
    each y_interest from interests
    each n_test from 0..9
    output:
    set y_interest, base_name, file(score_names), file(proba_names) into probability_slide
    file "*.png"

    script:
    compo = path.baseName
    base_name = compo + "_" + model + "_" + y_interest + "_" + "$lr" + "_" + "$n_test"
    weight_names = base_name + ".h5"
    score_names = base_name + ".pkl"
    proba_names = base_name + "_proba.csv"
    if( model == "resnet50convflat" )
        conv = 0
    else if( model == "resnet50convflatfreeze" )
        conv = 0
    else
        conv = 1
    """
    module load cuda10.0 
    python $tvt     --batch_size $batch_size \\
                    --path  $path\\
                    --labels $labels \\
                    --y_interest $y_interest \\
                    --out_weight  $weight_names\\
                    --model $model \\
                    --epochs $epochs \\
                    --dropout $dropout \\
                    --repeat 5 \\
                    --inner_fold $inner_fold \\
                    --multiprocess 0 \\
                    --workers 10 \\
                    --optimizer adam \\
                    --fold_test $n_test \\
                    --lr $lr \\
                    --filename $score_names \\
                    --probaname $proba_names \\
                    --fully_conv $conv
    """
}