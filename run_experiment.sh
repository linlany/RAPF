#!bin/bash

python main.py \
    --config-path configs/class \
    --config-name imagenet100_10-10.yaml \
    dataset_root="[imagenet_1k_path]" \
    class_order="class_orders/imagenet100.yaml"

python main.py \
    --config-path configs/class \
    --config-name imagenet_r_20-20.yaml \
    dataset_root="[imagenet_r_path]" \
    class_order="class_orders/imagenet_R_order.yaml"

python main.py \
    --config-path configs/class \
    --config-name cifar100_10-10.yaml \
    dataset_root="[cifar100_path]" \
    class_order="class_orders/cifar100_order.yaml"
