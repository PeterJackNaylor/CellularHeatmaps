
def get_options():
    import argparse
    parser = argparse.ArgumentParser(
        description='Creating training on heatmaps')

    parser.add_argument('--classes', required=False,
                        default=2,
                        metavar="int", type=int,
                        help='number of classes')

    parser.add_argument('--batch_size', required=False,
                        default=4,
                        metavar="int", type=int,
                        help='batch size value')

    parser.add_argument('--path', required=False,
                        default="/Users/naylorpeter/tmp/predict_from_umap_cell/patients/comp3",
                        metavar="str", type=str,
                        help='path to input data')

    parser.add_argument('--labels', required=False,
                        default="/Users/naylorpeter/tmp/predict_from_umap_cell/patients/multi_class.csv",
                        metavar="str", type=str,
                        help='path to label csv file')

    parser.add_argument('--y_interest', required=False,
                        default="RCB_class",
                        metavar="str", type=str,
                        help='tag for the variable of interest in labels')

    parser.add_argument('--loss', required=False,
                        default="categorical_crossentropy",
                        metavar="str", type=str,
                        help='type of loss, categorical_crossentropy or mse')

    parser.add_argument('--out_weight', required=False,
                        default="weight_models.h5",
                        metavar="str", type=str,
                        help='name for the weight file')   

    parser.add_argument('--model', required=False,
                        default='resnet50',
                        metavar="str", type=str,
                        help='name for the weight file')

    parser.add_argument('--epochs', required=False,
                        default=2,
                        metavar="int", type=int,
                        help='number of epochs')

    parser.add_argument('--workers', required=False,
                        default=5,
                        metavar="int", type=int,
                        help='number of workers')

    parser.add_argument('--repeat', required=False,
                        default=5,
                        metavar="int", type=int,
                        help='number of repeats for the evaluation of validation or test')

    parser.add_argument('--multiprocess', required=False,
                        default=0,
                        metavar="int", type=int,
                        help='wether to use multi_processing for keras')

    parser.add_argument('--inner_fold', required=False,
                        default=5,
                        metavar="int", type=int,
                        help='number of inner folds to perform')

    parser.add_argument('--fold_validation', required=False,
                        default=0,
                        metavar="int", type=int,
                        help='inner fold number to perform validation on')

    parser.add_argument('--fold_test', required=False,
                        default=0,
                        metavar="int", type=int,
                        help='outer fold number to perform test on')

    parser.add_argument('--optimizer', required=False,
                        default='adam',
                        metavar="str", type=str,
                        help='optimizer key')

    parser.add_argument('--lr', required=False,
                        default=10e-4,
                        metavar="float", type=float,
                        help='initial learning_rate')

    parser.add_argument('--callback', required=False,
                        default="version1",
                        metavar="str", type=str,
                        help='callback version to load')

    parser.add_argument('--dropout', required=False,
                        default=0.5,
                        metavar="float", type=float,
                        help='dropout value')

    parser.add_argument('--filename', required=False,
                        default="results.pkl",
                        metavar="str", type=str,
                        help='file name for val and test')


    parser.add_argument('--probaname', required=False,
                        default="proba.csv",
                        metavar="str", type=str,
                        help='proba file name for val and test')

    parser.add_argument('--fully_conv', required=False,
                        default=1,
                        metavar="int", type=int,
                        help='if it is fully conv at the end')
    args = parser.parse_args()
    args.fully_conv = args.fully_conv == 1
    return args

