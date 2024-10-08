#!/bin/bash

dataset_name=$1

wget -q -P dataset/beir/original https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/${dataset_name}.zip
unzip -q -d dataset/beir/original dataset/beir/original/${dataset_name}.zip
rm dataset/beir/original/${dataset_name}.zip