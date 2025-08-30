#!/bin/bash

# run all models for all catchments
for catchment in data/shapefiles/*.geojson; do
    catchment_name=$(basename $catchment .geojson)
    echo "Running models for ${catchment_name}"
    pixi run python3 ird_model/models/run_models.py "ird_model/models/inputs/config/${catchment_name}.toml" --stages $@
done