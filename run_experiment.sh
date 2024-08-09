#!bin/bash

python main.py \
    --config-path configs/class \
    --config-name imagenet100_10-10.yaml \
    dataset_root="[imagenet_path]" \
    class_order="class_orders/imagenet100.yaml"

