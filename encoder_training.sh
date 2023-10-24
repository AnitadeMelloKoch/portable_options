#!/bin/bash
python -m experiments.minigrid.advanced_doorkey.train_encoder --feature_size 500
python -m experiments.minigrid.advanced_doorkey.train_encoder --feature_size 200
python -m experiments.minigrid.advanced_doorkey.train_encoder --feature_size 400
python -m experiments.minigrid.advanced_doorkey.train_encoder --feature_size 300