#!/usr/bin/bash
# to set the directory to this one.
cd "$(dirname "$0")"

singularity build CellularHeatmaps.sif CellularHeatmaps.def