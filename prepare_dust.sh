#!/bin/bash

set -e

raw_dataset_path=$1
module_root_path=$2
operate_mode=$3

echo "raw dataset: $raw_dataset_path, module: $module_root_path"

create_dust_dir() {
    echo "create dust directory"
    cd $module_root_path
    cd data
    mkdir dust
    cd dust
    mkdir calib
    ln -s $raw_dataset_path/annotation/kitti_imagecamera ./label
    ln -s $raw_dataset_path/pcd ./pcd
    ln -s $raw_dataset_path/image_camera ./image
}

prepare_dust() {
    python main.py \
        --save_log, \
        --check_label, \
        --convert_datasets, \
        --set_split, \
        --raw_dataset_path $raw_dataset_path \
        --module_root_path $module_root_path \
        --train_split_rate 0.7 \
        --pcd_rect_rotate_angle 56
}

if [ $operate_mode == "create_dust_dir" ]; then
    create_dust_dir
elif [ $operate_mode == "prepare_dust" ]; then
    prepare_dust
else
    echo "operate mode error"
fi