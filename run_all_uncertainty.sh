#!/bin/bash

# run uncertainty models for all catchments
for catchment in data/shapefiles/*.geojson; do
    catchment_name=$(basename $catchment .geojson)
    echo "Running uncertainty models for ${catchment_name}"
    pixi run python3 ird_model/models/uncertainty.py "ird_model/models/inputs/config/${catchment_name}.toml"
done
