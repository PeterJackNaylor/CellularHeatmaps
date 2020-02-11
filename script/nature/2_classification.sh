for type in U2MAP U2MAP_reposition U3MAP U3MAP_reposition
do
    for res in 7 8 9
    do
        echo "####################################################################"
        echo 
        echo "########### Doing ${type} at ${res} ###############"
        echo 
        echo "####################################################################"
        nextflow run nextflow/Classification.nf -resume -c ~/.nextflow/config -profile mines \
                                        --PROJECT_NAME nature --PROJECT_VERSION 1-0 \
                                        --type $type --resolution $res \
                                        --label /mnt/data3/pnaylor/CellularHeatmaps/outputs/label_nature.csv
    done
done

