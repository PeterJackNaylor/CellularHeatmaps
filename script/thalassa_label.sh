python python/label_standardise/substrat_label.py --input_table ../Data/Biopsy_csv/RCB_substrat2.csv \
                                                  --folder_to_check ../Data/Biopsy_guillaume \
                                                  --output_table ../Data/Biopsy_csv/rcb_substrat.csv
python python/label_standardise/ftnbc_label.py --input_table ../Data/Biopsy_csv/MarickDataSet.xlsx \
                                               --folder_to_check ../Data/Biopsy/ \
                                               --output_table ../Data/Biopsy_csv/ftnbc.csv
python python/label_standardise/create_test_folds.py --substra ../Data/Biopsy_csv/rcb_substrat.csv \
                                                     --ftnbc ../Data/Biopsy_csv/ftnbc.csv \
                                                     --output_name ./outputs/label.csv \
                                                     --folder ../Data/nature_medecine_biop
