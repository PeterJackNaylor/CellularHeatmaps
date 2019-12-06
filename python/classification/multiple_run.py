
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
    number_validation = options.fold_validation
    test_fold = options.fold_test
    one_hot_encoding = True

    # outputs
    classes = options.classes if options.classes else 2
    met = options.loss

    metrics_names_with_loss = ['loss'] + metric_names
    metric_names = ["acc", "recall", "precision", "f1", "auc"]
    var = options.y_interest
    # options.loss == "categorical_crossentropy"

    ####### Hyper parameter model
    opt = options.optimizer
    lr = options.lr 
    batch_size = options.batch_size
    model_name = options.model
    fully_conv = options.fully_conv


    ####### output names
    weight_file = options.out_weight ## model weights to save
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
    dgi.create_inner_fold(inner_fold, test_fold)
    class_weight = None if options.fully_conv else dgi.return_weights() # doesn't work in fully convolutionnal

    dg_test = DataGenerator(dgi, size=(224,224,3), batch_size=1, 
                       shuffle=False, split='test', number=number_validation,
                       one_hot_encoding=one_hot_encoding, 
                       fully_conv=fully_conv, classes=classes)
    test_index = dgi.return_fold('test', 0)




    ####### Training


    probability = pd.DataFrame(index=test_index)

    all_results_dic = {}
    for n_val in range(inner_fold):

        ### setting up datagenerator for inner fold training
        dg_trai = DataGenerator(dgi, size=(224,224,3), batch_size=batch_size, 
                            shuffle=True, split='train', number=n_val,
                            one_hot_encoding=one_hot_encoding, 
                            fully_conv=fully_conv, classes=classes)

        dg_vali = DataGenerator(dgi, size=(224,224,3), batch_size=1, 
                            shuffle=False, split='validation', number=n_val,
                            fully_conv=fully_conv, one_hot_encoding=one_hot_encoding, classes=classes)

        # model prepration   
        model = load_model(model_name, classes=classes, dropout=options.dropout, batch_size=batch_size)
        optimizer = load_optimizer(opt, lr)
        model.compile(loss=met, optimizer=optimizer, metrics=import_metrics(met, classes))
      
        # training on inner_fold
        history = model.fit_generator(dg_trai, steps_per_epoch=len(dg_trai),
                                       epochs=epochs,  callbacks=call_backs,
                                       validation_data=dg_vali, 
                                       validation_steps=repeat * len(dg_vali),
                                       max_queue_size=10, workers=workers, 
                                       class_weight=class_weight,
                                       use_multiprocessing=multi_processing)

        if plot:
            import matplotlib as mpl
            mpl.use('Agg')
            import matplotlib.pyplot as plt
            for prefix in ["", "val"]:
                for meti in metric_names:
                    tag = '{}_{}'.format(prefix, meti) if prefix == "val" else meti
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
        output = multiple_test(dg_vali, steps=len(dg_vali), 
                                dg_test=dg_test, test_steps=len(dg_test),
                                repeat=repeat,
                                callbacks=call_backs, workers=workers,
                                use_multiprocessing=multi_processing,
                                metric_names=metrics_names_with_loss)
        vali_scores, val_dic, test_scores, test_dic = output
        import pdb; pdb.set_trace()
        all_results_dic.update(val_dic)
        all_results_dic.update(test_dic)

        if stdOut_print:
            ## print to standard output for fold progress.
            if len(metrics_names_with_loss) < 3:
                print("On validation {} we get:\n loss     | {} \n Accuracy | {}".format(n_val, *vali_scores))
                print("On test we get:\n loss     | {} \n Accuracy | {}".format(*test_scores))
            else:
                print("On validation {} we get:\n loss     | {} \n Accuracy | {}\n Recall   | {}\n precision| {}\n f1       |{}".format(n_val, *vali_scores))
                print("On test we get:\n loss     | {} \n Accuracy | {}\n Recall   | {}\n precision| {}\n f1       |{}".format(*test_scores))

    if save:
        output = open(filename, 'wb')
        pickle.dump(all_results_dic, output)
        output.close()
        probability.to_csv(options.probaname)

if __name__ == '__main__':
    main()
