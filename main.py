from pathlib import Path
import numpy as np
import os
import argparse
import datetime

from dust.dust_dataset import DustDataset
from dust.utils import common_util
from dust.prepare_dataset import PrepareDataset
from kitti.kitti_dataset import KittiDataset
from mine.mine_dataset import MineDataset

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--save_log', action='store_true', default=False, help='whether to save log file')
    parser.add_argument('--test_data', action='store_true', default=False, help='test something for datasets')
    parser.add_argument('--kitti', action='store_true', default=False, help='do work for kitti datasets')
    parser.add_argument('--mine', action='store_true', default=False, help='do work for mine datasets')
    parser.add_argument('--draw_bbox', action='store_true', default=False, help='draw bbox in pcd image')
    parser.add_argument('--check_label', action='store_true', default=False, help='whether to check label')
    parser.add_argument('--convert_datasets', action='store_true', default=False, help='convert datasets from pcd to bin')
    parser.add_argument('--set_split', action='store_true', default=False, help='set split sample to train and value')
    parser.add_argument('--box_dist_statis', action='store_true', default=False, help='all box distribution statistic')
    parser.add_argument('--raw_dataset_path', type=str, default="", help='specify the datasets path to convert pcd to bin')
    parser.add_argument('--module_root_path', type=str, default="", help='specify the module root path')
    parser.add_argument('--train_split_rate', type=float, default=0.7, help='the rate of split train datasets')
    parser.add_argument('--pcd_rect_rotate_angle', type=int, default=0, help='the rotate angle to rect pcd to head')
    args = parser.parse_args()
    
    return args
    
def create_logger(args):
    log_file = None
    if (args.save_log):
        log_dir = Path(os.getcwd()) / 'log'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / ('log_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    return common_util.create_logger(log_file)
    
def test_dust(logger, args):
    data_path = Path(args.raw_dataset_path)
    dust_dataset = DustDataset(dataset_dir=data_path)
    dust_dataset
    
    data_ids = [ '1661512340_793590', '1661512370_697764', 
                '1661512376_400516',
                '1661512367_398879',
                '1661512357_099283',
                '1661512356_210098',
                '1661512384_901863',
                '1661512379_404696'
    ]
    check_id = 0
    
    # dust_dataset.get_anchor_size(Path(r'C:\Users\Y\Documents\dust_datasets\2022-08-26-19-12-20\annotation\kitti_imagecamera'))
    # dust_dataset.convert_pcd_to_bin(data_ids[check_id])
    # lidar_a = dust_dataset.get_lidar_by_idx(data_ids[check_id])
    # lidar_b = dust_dataset.get_lidar_bin_by_idx(data_ids[check_id])
    # dust_dataset.visualize_point_cloud_by_idx(data_ids[check_id])
    dust_dataset.visualize_3d_bbox_center_in_image(data_ids[check_id])
    dust_dataset.visualize_3d_bbox_in_image(data_ids[check_id])
    
    x_range = (5, 500)
    # y_range = (-6.4, 0.8)
    y_range=(-50, 60)
    z_range = (-5, 5)
    rotate_angle = 56
    dust_dataset.visualize_point_cloud_by_idx_and_angle(data_ids[check_id], rotate_angle)
    # dust_dataset.visualize_point_cloud_by_idx_and_angle_and_filter(data_ids[check_id], rotate_angle, x_range, y_range, z_range)
    dust_dataset.draw_3d_bbox_in_rect_point_cloud_by_index(data_ids[check_id], rotate_angle, x_range, y_range, z_range)
    # dust_dataset.visualize_rotate_and_filter_point_cloud_in_image(data_ids[check_id], rotate_angle, x_range, y_range, z_range)
    
    point_cloud_range = np.array([5, -6.4, -4, 53.8, 0.8, 0])
    # dust_dataset.check_box_outside_range(
    #     logger=logger,
    #     label_path=Path(r'C:\Users\Y\Documents\dust_datasets\2022-08-26-19-12-20\annotation\kitti_imagecamera'), 
    #     limit_range=point_cloud_range, 
    #     rotate_angle=rotate_angle
    # )
    
    x_range = (2, 20)
    y_range = (-20, -5)
    z_range = (-3, 2)
    # dust_dataset.visualize_point_cloud_by_idx_and_filter(data_ids[check_id], x_range, y_range, z_range)
    # dust_dataset.draw_3d_bbox_in_filter_point_cloud_by_index(data_ids[check_id], x_range, y_range, z_range)
    # dust_dataset.visualize_filter_point_cloud_in_fov(data_ids[check_id], x_range, y_range, z_range)
    dust_dataset.visualize_filter_point_cloud_in_image(data_ids[check_id], x_range, y_range, z_range)
    
def main():
    args = parse_config()
    logger = create_logger(args)

    # from taizhong import dataset_manager
    # dataset_manager.normalize_pcd_format_from_editor(logger)
    
    if (args.test_data):
        test_dust(logger, args)
    elif (args.kitti):
        kittiDataset = KittiDataset(logger, args)
        kittiDataset.count_labels()
    elif (args.mine):
        mineDataset = MineDataset(logger, args)
        mineDataset.count_labels()
    else:
        prepareDataset = PrepareDataset(logger, args)
        prepareDataset.process_raw_dataset(args)

if __name__=='__main__':
    main()