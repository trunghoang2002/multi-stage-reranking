#!/bin/bash

dataset_name=$1

python preprocess_beir.py \
--data_path dataset/beir/original/${dataset_name} \
--output_data_path dataset/beir/processed/${dataset_name} \
--model_name_or_path FacebookAI/roberta-large

python preprocess_bm25.py \
--data_path dataset/beir/processed/${dataset_name} \
--output_data_path dataset/beir/processed_bm25/${dataset_name}
