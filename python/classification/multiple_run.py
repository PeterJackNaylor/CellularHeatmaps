
import os

import numpy as np
import pickle
import pandas as pd 

from keras_object import DataGenerator
from metrics import import_metrics
from models import load_model, call_backs_load, load_optimizer
from data_object import DataGenImage

from options_multiple_run import get_options

from multiple_test import multiple_test 

def main():

    #### General options
    plot = True
    save = True
    stdOut_print = True
    options = get_options()

    #### Training hyper-parameters
    inner_fold = options.inner_fold
    fold_test = options.fold_test
    one_hot_encoding = True

    # outputs
    classes = options.classes if options.classes else 2
    met = options.loss
    
    metric_names = ["acc", "recall", "precision", "f1", "auc_roc"]
    metrics_names_with_loss = ['loss'] + metric_names
    var = options.y_interest
    # options.loss == "categorical_crossentropy"

    ####### Hyper parameter model
    opt = options.optimizer
    lr = options.lr 
    batch_size = options.batch_size
    model_name = options.model


    ####### output names
    weight_file = options.out_weight ## model weights to save
    ## Note the weight file is just a temporary file to reload the best model after..
    filename = options.filename  ## dictionnary of results filename

    ####### Hyper parameter model general
    epochs = options.epochs
    workers = options.workers
    

    multi_processing = options.multiprocess == 1
    callback_version = options.callback
    repeat = options.repeat

    call_backs = call_backs_load(callback_version, weight_file)



    ###### Data generator

    path = options.path
    labels_path = options.labels

    dgi = DataGenImage(path, labels_path, var, classes=classes)
    dgi.create_inner_fold(inner_fold, fold_test)
    class_weight = dgi.return_weights() # doesn't work in fully convolutionnal

    dg_test = DataGenerator(dgi, size=(224,224,3), batch_size=1, 
                       shuffle=False, split='test',
                       one_hot_encoding=one_hot_encoding, 
                       classes=classes)
    test_index = dgi.return_fold('test', 0)




    ####### Training

    all_results_dic = {}
    for n_val in range(inner_fold):

        ### setting up datagenerator for inner fold training
        dg_train = DataGenerator(dgi, size=(224,224,3), batch_size=batch_size, 
                            shuffle=True, split='train', number=n_val,
                            one_hot_encoding=one_hot_encoding, 
                            classes=classes)

        dg_validation = DataGenerator(dgi, size=(224,224,3), batch_size=1, 
                            shuffle=False, split='validation', number=n_val,
                            one_hot_encoding=one_hot_encoding, classes=classes)

        # model prepration   
        model = load_model(model_name, classes=classes, dropout=options.dropout, batch_size=batch_size)
        optimizer = load_optimizer(opt, lr)
        model.compile(loss=met, optimizer=optimizer, metrics=import_metrics(met, classes))
      
        # training on inner_fold
        history = model.fit_generator(dg_train, steps_per_epoch=len(dg_train),
                                       epochs=epochs,  callbacks=call_backs,
                                       validation_data=dg_validation, 
                                       validation_steps=repeat * len(dg_validation),
                                       max_queue_size=10, workers=workers, 
                                       class_weight=class_weight,
                                       use_multiprocessing=multi_processing)

        if plot:
            import matplotlib as mpl
            mpl.use('Agg')
            import matplotlib.pyplot as plt
            for prefix in ["", "val"]:
                for metric in metric_names:
                    tag = '{}_{}'.format(prefix, metric) if prefix == "val" else metric
                    plt.plot(range(epochs), history.history[tag], label=tag)
            plt.legend()
            plt.savefig("training_curves_{}.png".format(n_val))
            plt.close()
            plt.plot(range(epochs), history.history['loss'], label='loss')
            plt.plot(range(epochs), history.history['val_loss'], label='val_loss')
            plt.savefig("training_loss_curves_{}.png".format(n_val))
            plt.close()
        
        # restore best weights from model, callback
        model.load_weights(weight_file)
        # Maybe something to do to improve robustness, take max over patients or something.
        output = multiple_test(dg_validation, steps=len(dg_validation), 
                               dg_test=dg_test, test_steps=len(dg_test),
                               repeat=repeat,
                               callbacks=call_backs, workers=workers,
                               use_multiprocessing=multi_processing,
                               metric_names=metrics_names_with_loss,
                               model=model,
                               fully_conv=False)

        validation_scores, val_prob, val_variance, test_scores, test_prob, test_variance = output
        validation_index = dgi.return_fold("validation", n_val)
        validation_prob_df = pd.DataFrame(val_prob, index=validation_index)
        test_prob_df = pd.DataFrame(test_prob, index=test_index)

        for i in range(val_prob.shape[1]):
            validation_prob_df["{}_v".format(i)] = val_variance[:,i]
            test_prob_df["{}_v".format(i)] = test_variance[:,i]
        
        result_run = {'{}_validation'.format(n_val):validation_scores, 
                      '{}_validation_prob'.format(n_val):validation_prob_df,
                      '{}_test'.format(n_val):test_scores, 
                      '{}_test_prob'.format(n_val):test_prob_df}

        all_results_dic.update(result_run)

        if stdOut_print:
            ## print to standard output for fold progress.
            if len(metrics_names_with_loss) < 3:
                print("On validation {} we get:\n loss     | {} \n Accuracy | {}".format(n_val, *vali_scores))
                print("On test we get:\n loss     | {} \n Accuracy | {}".format(*test_scores))
            else:
                dv = validation_scores
                print("On validation {} we get:\n loss     | {} \n Accuracy | {}\n Recall   | {}\n precision| {}\n f1       | {}\n roc-auc  | {}".format(n_val, dv['loss'], dv['acc'], dv['recall'], dv['precision'], dv['f1'], dv['auc_roc']))
                print("On test we get:\n loss     | {} \n Accuracy | {}\n Recall   | {}\n precision| {}\n f1       | {}\n roc-auc  | {}".format(test_scores['loss'], test_scores['acc'], test_scores['recall'], test_scores['precision'], test_scores['f1'], test_scores['auc_roc']))

    if save:
        output = open(filename, 'wb')
        pickle.dump(all_results_dic, output)
        output.close()

if __name__ == '__main__':
    main()
