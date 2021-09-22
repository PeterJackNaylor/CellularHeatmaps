

ENV=export PYTHONPATH=`pwd`/src/templates/segmentation:$${PYTHONPATH}


image:
	sudo bash environment/create-img.sh

test_run: src/CellularSegmentation.nf
	$(ENV); nextflow $< -resume -c nextflow.config -profile local \
						--PROJECT_NAME local --PROJECT_VERSION 1-0 \
						--tiff_location "/data/dataset/camelyon2016/*/*/*.tif" \
						--nucleus_segmentation_model ../segmentation-he/outputs/nuclei_segmentation_model_with_aji/nuclei_segmentation_model \
						--tissue_segmentation_model ../segmentation-he/outputs/tissue_segmentation_model \
						--save 1

camelyon: src/CellularSegmentation.nf
	$(ENV); nextflow $< -resume -c nextflow.config -profile kuma \
						--PROJECT_NAME kuma --PROJECT_VERSION 1-0 \
						--tiff_location "/data2/pnaylor/datasets/pathology/Camelyon2016/*/*/*.tif" \
						--nucleus_segmentation_model /data2/pnaylor/models/nuclei_segmentation_model \
						--tissue_segmentation_model /data2/pnaylor/models/tissue_segmentation_model \
						--save 0


clean:
	nextflow clean
	# maybe remove singularity image and clean up...?