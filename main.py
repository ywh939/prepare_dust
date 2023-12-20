from dust.dust_dataset import DustDataset
from pathlib import Path
import numpy as np
import os

def test_dust():
    data_path = Path(r'C:\Users\Y\Documents\dust_datasets\2022-08-26-19-12-20')
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
    check_id = 7
    
    # dust_dataset.get_anchor_size(Path(r'C:\Users\Y\Documents\dust_datasets\2022-08-26-19-12-20\annotation\kitti_imagecamera'))
    dust_dataset.convert_pcd_to_bin(data_ids[check_id])
    # lidar_a = dust_dataset.get_lidar_by_idx(data_ids[check_id])
    # lidar_b = dust_dataset.get_lidar_bin_by_idx(data_ids[check_id])
    # dust_dataset.visualize_point_cloud_by_idx(data_ids[check_id])
    # dust_dataset.visualize_3d_bbox_center_in_image(data_ids[check_id])
    # dust_dataset.visualize_3d_bbox_in_image(data_ids[check_id])

    x_range = (5, 60)
    y_range = (-6.4, 0.8)
    z_range = (-5, 0)
    rotate_angle = 56
    # dust_dataset.visualize_point_cloud_by_idx_and_angle(data_ids[check_id], rotate_angle)
    # dust_dataset.visualize_point_cloud_by_idx_and_angle_and_filter(data_ids[check_id], rotate_angle, x_range, y_range, z_range)
    # dust_dataset.draw_3d_bbox_in_rect_point_cloud_by_index(data_ids[check_id], rotate_angle, x_range, y_range, z_range)
    # dust_dataset.visualize_rotate_and_filter_point_cloud_in_image(data_ids[check_id], rotate_angle, x_range, y_range, z_range)
    
    point_cloud_range = np.array([5, -6.4, -4, 53.8, 0.8, 0])
    # dust_dataset.check_box_outside_range(Path(r'C:\Users\Y\Documents\dust_datasets\2022-08-26-19-12-20\annotation\kitti_imagecamera'), point_cloud_range, rotate_angle)
    
    x_range = (2, 20)
    y_range = (-20, -5)
    z_range = (-3, 2)
    # dust_dataset.visualize_point_cloud_by_idx_and_filter(data_ids[check_id], x_range, y_range, z_range)
    # dust_dataset.draw_3d_bbox_in_filter_point_cloud_by_index(data_ids[check_id], x_range, y_range, z_range)
    # dust_dataset.visualize_filter_point_cloud_in_fov(data_ids[check_id], x_range, y_range, z_range)
    # dust_dataset.visualize_filter_point_cloud_in_image(data_ids[check_id], x_range, y_range, z_range)
    
def convert_kitti(data_path):
    dust_dataset = DustDataset(dataset_dir=data_path)
    file_list = os.listdir(dust_dataset.lidar_dir)
    for file in file_list:
        file_split = os.path.splitext(file)
        file_type = file_split[1]
        if (file_type != '.pcd'):
            continue
        
        file_name = file_split[0]
        bin_file = file_name + '.bin'
        if (bin_file in file_list):
            continue
        
        dust_dataset.convert_pcd_to_bin(file_name)

def main():
    # test_dust()
    convert_kitti(Path(r'C:\Users\Y\Documents\dust_datasets\2022-08-26-19-12-20'))

if __name__=='__main__':
    main()