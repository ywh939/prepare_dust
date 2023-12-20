from pathlib import Path
import numpy as np
import os
import argparse

from dust.dust_dataset import DustDataset

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--test_data', action='store_true', default=False, help='test something for datasets')
    parser.add_argument('--convert_datasets', action='store_true', default=False, help='convert datasets from pcd to bin')
    parser.add_argument('--set_split', action='store_true', default=False, help='set split sample to train and value')
    parser.add_argument('--convert_datasets_path', type=str, default="", help='specify the datasets path to convert pcd to bin')
    parser.add_argument('--module_root_path', type=str, default="", help='specify the module root path')
    args = parser.parse_args()
    
    return args
    
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
    data_path = Path(data_path)
    assert data_path.exists()
    print(f"the path of datasets to convert pcd to bin: {data_path}")
    
    dust_dataset = DustDataset(dataset_dir=data_path)
    file_list = os.listdir(dust_dataset.lidar_dir)
    exist_bin_file_count = 0
    converted_count = 0
    for file in file_list:
        file_split = os.path.splitext(file)
        file_type = file_split[1]
        if (file_type != '.pcd'):
            continue
        
        file_name = file_split[0]
        bin_file = file_name + '.bin'
        if (bin_file in file_list):
            exist_bin_file_count += 1
            continue
        
        dust_dataset.convert_pcd_to_bin(file_name)
        converted_count += 1
    
    print(f'exist {exist_bin_file_count} bin files, converted {converted_count} files')
    
def save_split_sample(sample_list, save_path):
    with open(save_path, 'w') as file:
        file.write('\n'.join(sample_list))
    print(f'save split sample {len(sample_list)} to {save_path}')
    
def set_split_datasest(data_path, module_root_path):
    data_path = Path(data_path)
    assert data_path.exists()
    module_root_path = Path(module_root_path)
    assert module_root_path.exists()
    
    print(f'start split datasets: {data_path}')
    
    dust_dataset = DustDataset(dataset_dir=data_path)
    file_list = os.listdir(dust_dataset.lidar_dir)
    bin_data_list = list(filter(lambda x: x.endswith('.bin'), file_list))
    data_list = [s.replace('.bin', '') for s in bin_data_list]
    
    if len(data_list) == 0:
        print('no bin files')
        return
    
    from sklearn.model_selection import train_test_split
    train_list, val_list = train_test_split(data_list, test_size=0.3, random_state=42)

    print(f'total data: {len(data_list)}, train data: {len(train_list)}, X_val: {len(val_list)}')
    
    save_split_path = module_root_path / 'data' / 'dust' / 'split'
    save_split_path.mkdir(parents=True, exist_ok=True)
    
    train_split_path = save_split_path / 'train.txt'
    save_split_sample(train_list, train_split_path)
    
    val_split_path = save_split_path / 'val.txt'
    save_split_sample(val_list, val_split_path)

def main():
    args = parse_config()
    
    if (args.test_data):
        test_dust()
    
    if (args.convert_datasets):
        convert_kitti(args.convert_datasets_path)
    
    if (args.set_split):
        set_split_datasest(args.convert_datasets_path, args.module_root_path)

if __name__=='__main__':
    main()