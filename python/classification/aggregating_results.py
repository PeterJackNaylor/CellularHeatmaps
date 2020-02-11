import pandas as pd
import pickle as pkl
from glob import glob
import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd 

inner_fold = 5
label_file = "/mnt/data3/pnaylor/CellularHeatmaps/outputs/label_nature.csv"
y_interest = "Residual"


def get_options():
    import argparse
    parser = argparse.ArgumentParser(
        description='Creating training on heatmaps')
    parser.add_argument('--labels', required=False,
                        default="/Users/naylorpeter/tmp/predict_from_umap_cell/patients/multi_class.csv",
                        metavar="str", type=str,
                        help='path to label csv file')
    parser.add_argument('--y_interest', required=False,
                        default="RCB_class",
                        metavar="str", type=str,
                        help='tag for the variable of interest in labels')
    parser.add_argument('--inner_fold', required=False,
                        default=5,
                        metavar="int", type=int,
                        help='number of inner folds to perform')
    parser.add_argument('--filename', required=False,
                        default="results.csv",
                        metavar="str", type=str,
                        help='file name for val and test')
    args = parser.parse_args()
    return args


def main():

    options = get_options()
    inner_fold = options.inner_fold
    label_file = options.labels
    y_interest = options.y_interest


    label = pd.read_csv(label_file, index_col="Biopsy")[y_interest]
    list_dic = []
    for f in glob("*.pkl"):
        dic = pkl.load(open(f, 'rb'))
        validation_predictions = [dic["{}_validation_prob".format(i)].join(label) for i in range(inner_fold)]
        test_predictions = [dic["{}_test_prob".format(i)].join(label) for i in range(inner_fold)]

        auc_scores = []
        auc_scores_t = []
        for i in range(inner_fold):
            y_scores = validation_predictions[i][1]
            y_true = validation_predictions[i][y_interest]
            auc_scores.append(roc_auc_score(y_true, y_scores))

            y_scores_t = test_predictions[i][1]
            y_true_t = test_predictions[i][y_interest]
            auc_scores_t.append(roc_auc_score(y_true_t, y_scores_t))

        best_ind = np.argmax(auc_scores)
        auc_score_best_val = auc_scores[best_ind]
        auc_score_best_val_t = auc_scores_t[best_ind]


        validation_predictions = pd.concat(validation_predictions, axis=0)
        y_scores = validation_predictions[1]
        y_true = validation_predictions[y_interest]
        avg_auc = roc_auc_score(y_true, y_scores)


        test_predictions_c = pd.concat(test_predictions, axis=0)
        y_scores_t = test_predictions_c[1]
        y_true_t = test_predictions_c[y_interest]
        avg_auc_t = roc_auc_score(y_true_t, y_scores_t)

        res = {'max_val': auc_score_best_val,
               'max_val_t': auc_score_best_val_t,
               'max_val_t_p': test_predictions[best_ind],
               'avg_val': avg_auc,
               'avg_val_t': avg_auc_t,
               'avg_val_t_p': test_predictions_c}

        list_dic.append((f, dic, res))
    
    final_results = pd.DataFrame(index=range(len(list_dic)), columns=["fold", "model", "lr", "max_val"])
    ind = 0
    for name, dic, res in list_dic:
        _, _,  fold, _, model, _, lr = name.split('_')
        lr = lr.split('.p')[0]

        final_results.ix[ind, "fold"] = fold
        final_results.ix[ind, "model"] = model
        final_results.ix[ind, "lr"] = lr
        final_results.ix[ind, "max_val"] = res["max_val"]
        final_results.ix[ind, "mean_val"] = res["avg_val"]
        ind += 1
    
    final_results['max_val'] = final_results['max_val'].astype('float')

    gfr = final_results.groupby(["model", "lr"])
    final_final = gfr.mean()
    for g in gfr:
        avg_p = []
        max_p = []
        for n in range(g[1].shape[0]):
            ind_n = (g[1]).index[n]
            _, _, res = list_dic[ind_n]
            avg_p.append(res['avg_val_t_p'])
            max_p.append(res['max_val_t_p'])
        
        prob_t_avg = pd.concat(avg_p, axis=0)
        prob_t_max = pd.concat(max_p, axis=0)

        y_scores = prob_t_max[1]
        y_true = prob_t_max[y_interest]
        auc_max_test = roc_auc_score(y_true, y_scores)
        final_final.ix[g[0], 'test_max'] = auc_max_test

        y_scores = prob_t_avg[1]
        y_true = prob_t_avg[y_interest]
        auc_avg_test = roc_auc_score(y_true, y_scores)
        final_final.ix[g[0], 'test_avg'] = auc_avg_test

        final_final.to_csv(options.filename)

if __name__ == "__main__":
    main()