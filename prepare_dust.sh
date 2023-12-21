#!/bin/bash -e

raw_dataset_path=$1
module_root_path=$2
echo 'raw dataset: $raw_dataset_path, module: $module_root_path'

cd module_root_path
cd data
mkdir dust
cd dust
mkdir calib
ln -s $raw_dataset_path/annotation/kitti_imagecamera ./label
ln -s $raw_dataset_path/pcd ./pcd
ln -s $raw_dataset_path/image_camera ./image


