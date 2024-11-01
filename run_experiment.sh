#!bin/bash

python main.py \
    --config-path configs/class \
    --config-name imagenet100_10-10.yaml \
    dataset_root="/defaultShare/pubdata/imagenet/" \
    class_order="class_orders/imagenet100.yaml"

python main.py \
    --config-path configs/class \
    --config-name imagenet_r_20-20.yaml \
    dataset_root="/huanglinlan/back_code/imagenet-r-split/" \
    class_order="class_orders/imagenet_R_order.yaml"

python main.py \
    --config-path configs/class \
    --config-name cifar100_10-10.yaml \
    dataset_root="/defaultShare/archive/huanglinlan/" \
    class_order="class_orders/cifar100_order.yaml"
