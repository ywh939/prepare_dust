import os
from pathlib import Path
import numpy as np

from dust.dust_dataset import DustDataset
from dust.utils import common_util
from dust.utils import dust_util

class PrepareDataset(object):
    def __init__(self, logger, args):
        raw_dataset_path = Path(args.raw_dataset_path)
        assert raw_dataset_path.exists()
        module_root_path = Path(args.module_root_path)
        assert module_root_path.exists()
        rotate_angle = args.pcd_rect_rotate_angle        
        # the number 56 is the rotate angle of the lidar, that can rectify the x coordinate of
        # lidar point cloud to the heading direction of the vehicle
        assert rotate_angle != 0
        
        self.train_split_rate = args.train_split_rate
        
        self.logger = logger
        self.raw_dataset_path = raw_dataset_path
        self.module_root_path = module_root_path
        self.rotate_angle = rotate_angle
        
        self.dust_dataset = DustDataset(dataset_dir=self.raw_dataset_path)
        
        self.statistic_log_path = self.dust_dataset.label_dir.parent
        self.invalid_label_path = self.statistic_log_path / 'invalid_label.txt'
        
    def process_raw_dataset(self, args):
        if (args.check_label):
            self.check_label()
        
        if (args.convert_datasets):
            self.convert_kitti()
        
        if (args.set_split):
            self.set_split_datasest()
            
    def check_label(self):
        self.logger.info(f"start to check label: {self.dust_dataset.label_dir}")
        
        label_list = os.listdir(self.dust_dataset.label_dir)
        obj_type = []
        all_corners = None
        invalid_label = []
        
        for label_name in label_list:
            objs = self.dust_dataset.get_label_objects_by_path(self.dust_dataset.label_dir / label_name)
            objs_num = len(objs)
            if objs_num == 0:
                self.logger.error(f'invalid objects num: {objs_num}')
                invalid_label.append(os.path.splitext(label_name)[0])
                continue
            
            for obj in objs:
                if obj.l <= 0 or obj.w <= 0 or obj.h <= 0:
                    self.logger.error(f"label {label_name}, object type {obj.type}, Invalid size: {obj.lwh}")
                    invalid_label.append(os.path.splitext(label_name)[0])
                    continue
                
                if obj.type not in obj_type:
                    obj_type.append(obj.type)
                    
                corners = self.get_rect_3dbox(obj)
                if all_corners is None:
                    all_corners = corners
                else:
                    all_corners = np.concatenate((all_corners, corners), axis=1)
                    
        self.logger.info(f'object type: {obj_type}')
        
        self.statistic_box_distribution(self.logger, all_corners)
        
        common_util.save_to_file_line_by_line(self.logger, invalid_label, self.invalid_label_path)
                    
    def get_rect_3dbox(self, obj):
        boxes = np.concatenate([obj.center, obj.lwh, np.asanyarray(obj.yaw_radian).reshape(-1)], axis=-1).reshape(1, -1)
        boxes[:, 0:3] = dust_util.rotate_lidar_along_z(boxes[:, 0:3], self.rotate_angle)
        boxes[:, 6] = dust_util.rect_to_yaw(float(boxes[:, 6]), self.rotate_angle)
        return dust_util.boxes_to_corners_3d(boxes)

    def statistic_box_distribution(self, logger, all_corners):
        all_box_x = all_corners[:, :, 0].flatten().tolist()
        all_box_y = all_corners[:, :, 1].flatten().tolist()
        all_box_z = all_corners[:, :, 2].flatten().tolist()
        
        save_path_root = self.statistic_log_path
        
        dust_util.draw_hist(data=all_box_x,
                            title=f'lidar box x coordinate distribution',
                            xlabel=f'x coordinate value',
                            ylabel='count',
                            save_path=save_path_root / f'x_coordinate_distribution.png'
                            )
        dust_util.draw_hist(data=all_box_y,
                            title=f'lidar box y coordinate distribution',
                            xlabel=f'y coordinate value',
                            ylabel='count',
                            save_path=save_path_root / f'y_coordinate_distribution.png'
                            )
        dust_util.draw_hist(data=all_box_z,
                            title=f'lidar box z coordinate distribution',
                            xlabel=f'z coordinate value',
                            ylabel='count',
                            save_path=save_path_root / f'z_coordinate_distribution.png'
                            )
        logger.info(f'save box distribution to {save_path_root}')
        
    def convert_kitti(self):
        self.logger.info(f"the path of datasets to convert pcd to bin: {self.raw_dataset_path}")
        
        file_list = os.listdir(self.dust_dataset.lidar_dir)
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
            
            self.dust_dataset.convert_pcd_to_bin(file_name)
            converted_count += 1
        
        self.logger.info(f'exist {exist_bin_file_count} bin files, converted {converted_count} files')
        
    def get_invalid_label(self):
        if (not os.path.exists(self.invalid_label_path)):
            return []
        
        with open(self.invalid_label_path, 'r') as f:
            lines = [x.strip() for x in f.readlines()]
        return lines
        
    def set_split_datasest(self):
        self.logger.info(f'start split datasets: {self.raw_dataset_path}')
        
        file_list = os.listdir(self.dust_dataset.lidar_dir)
        bin_data_list = list(filter(lambda x: x.endswith('.bin'), file_list))
        data_list = [s.replace('.bin', '') for s in bin_data_list]
        invalid_label = self.get_invalid_label()
        valid_list = common_util.delete_list_elem_obtain_other(data_list, invalid_label)
        
        if len(data_list) == 0:
            self.logger.error('no bin files')
            return
        
        from sklearn.model_selection import train_test_split
        train_list, val_list = train_test_split(valid_list, train_size=self.train_split_rate, random_state=42)

        self.logger.info(f'total data: {len(data_list)}, valid data: {len(valid_list)}, split rate: {self.train_split_rate}, train data: {len(train_list)}, X_val: {len(val_list)}')
        
        save_split_path = self.module_root_path / 'data' / 'dust' / 'split'
        save_split_path.mkdir(parents=True, exist_ok=True)
        
        train_split_path = save_split_path / 'train.txt'
        common_util.save_to_file_line_by_line(self.logger, train_list, train_split_path)
        
        val_split_path = save_split_path / 'val.txt'
        common_util.save_to_file_line_by_line(self.logger, val_list, val_split_path)
    