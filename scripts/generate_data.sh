#!/bin/bash

# Arrays of your dataset paths and config paths
DATA_PATHS=("../data/reactions/bh/bh_reaction_1.csv" "../data/reactions/bh/bh_reaction_2.csv" "../data/reactions/bh/bh_reaction_3.csv" "../data/reactions/bh/bh_reaction_4.csv" "../data/reactions/bh/bh_reaction_5.csv")
CONFIG_PATHS=("../templates/basic.yaml" "../templates/moderate.yaml" "../templates/full.yaml")

# Nested loop to iterate over all combinations of datasets and configs
for DATA_PATH in "${DATA_PATHS[@]}"; do
    for CONFIG_PATH in "${CONFIG_PATHS[@]}"; do
        # Run your script with the current combination of data path and config path
        python data_preprocessing.py --data_path "$DATA_PATH" --config_path "$CONFIG_PATH"
    done
done
