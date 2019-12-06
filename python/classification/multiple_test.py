import numpy as np        

def mult_test(dg, steps, repeat, callbacks, workers,
              use_multiprocessing,
              metric_names, model, fully_conv):
    
    scores = model.evaluate_generator(dg, steps=repeat * steps, 
                                      callbacks=callbacks, workers=workers,
                                      use_multiprocessing=use_multiprocessing) 

    # turn on data augmentation?
    prob = model.predict_generator(dg, steps=repeat * steps, 
                                   callbacks=callbacks, workers=workers,
                                   use_multiprocessing=use_multiprocessing)

    prob_final = np.zeros(steps, prob.shape[-1])
    for i in range(repeat):
        if fully_conv:
            prob_final += prob_final[i*steps:(i+1)*steps,0,0]
        else:
            prob_final += prob_final[i*steps:(i+1)*steps]
    prob_final /= repeat
    scores_d = {metric_names[i]:scores[i] for i in range(len(metric_names))}
    return scores_d, prob_final



def multiple_test(dg_vali, steps=0, 
                    dg_test=None, test_steps=0,
                    repeat=0,
                    callbacks=None, workers=1,
                    use_multiprocessing=False,
                    metric_names=None,
                    model=None,
                    fully_conv=None):

    val_s, val_prob = mult_test(dg_vali, steps=steps,
                                repeat=repeat,
                                callbacks=callbacks, workers=workers,
                                use_multiprocessing=use_multiprocessing,
                                metric_names=metric_names,
                                model=model,
                                fully_conv=fully_conv)       
    if test_steps:
        test_s, test_prob = mult_test(dg_test, steps=test_steps,
                                repeat=repeat,
                                callbacks=callbacks, workers=workers,
                                use_multiprocessing=use_multiprocessing,
                                metric_names=metric_names,
                                model=model,
                                fully_conv=fully_conv)       
    return val_s, val_prob, test_s, test_prob
       
 