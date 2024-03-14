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
        
        self.classes = ['Car']
        self.class_objs = {'Car':1}
        
    def process_raw_dataset(self, args):
        if (args.check_label):
            self.check_label()
        
        if (args.convert_datasets):
            self.convert_kitti()
        
        if (args.set_split):
            self.set_split_datasest()

        if (args.box_dist_statis):
            self.statistic_all_box_distribution()
            
    def check_label(self):
        self.logger.info(f"start to check label: {self.dust_dataset.label_dir}")
        
        label_list = os.listdir(self.dust_dataset.label_dir)
        invalid_label = self.check_invalid_label(label_list)
        
        all_corners = None
        obj_type = []
        for label_name in label_list:
            if os.path.splitext(label_name)[0] in invalid_label:
                continue
            objs = self.dust_dataset.get_label_objects_by_path(self.dust_dataset.label_dir / label_name)
            for obj in objs:
                if obj.type not in obj_type:
                    obj_type.append(obj.type)
                    
                corners = self.get_rect_3dbox(obj)
                if all_corners is None:
                    all_corners = corners
                else:
                    all_corners = np.concatenate((all_corners, corners), axis=1)
                    
        self.logger.info(f'object type: {obj_type}')
        
        self.statistic_box_distribution(all_corners, 'clean_data_box_dist')
        
        common_util.save_to_file_line_by_line(self.logger, invalid_label, self.invalid_label_path)
        
    def check_invalid_label(self, label_list):
        invalid_label = []
        
        for label_name in label_list:
            objs = self.dust_dataset.get_label_objects_by_path(self.dust_dataset.label_dir / label_name)
            objs_num = len(objs)
            if objs_num == 0:
                self.logger.error(f'invalid objects num: {objs_num}')
                invalid_label.append(os.path.splitext(label_name)[0])
                continue
            
            class_objs_cnt = {}
            for obj in objs:
                if obj.is_invalid():
                    self.logger.error(f"label {label_name}, object type {obj.type}, invalid: lwh={obj.lwh}, box2d={obj.box2d}")
                    invalid_label.append(os.path.splitext(label_name)[0])
                    break
                
                if obj.type not in self.classes:
                    self.logger.warning(f"label {label_name}, object type {obj.type}, not valid")
                    invalid_label.append(os.path.splitext(label_name)[0])
                    break
                
                val = class_objs_cnt.get(obj.type, 0) + 1
                class_objs_cnt[obj.type] = val
            
            for obj_type, objs_num in class_objs_cnt.items():
                if objs_num > self.class_objs[obj_type]:
                    self.logger.error(f"label {label_name}, object type {obj.type}, object num: {objs_num} is greater than limit: {self.class_objs[obj.type]}")
                    invalid_label.append(os.path.splitext(label_name)[0])
        
        self.logger.info(f"invalid label count: {len(invalid_label)}")
        return invalid_label
                    
    def get_rect_3dbox(self, obj):
        boxes = np.concatenate([obj.center, obj.lwh, np.asanyarray(obj.yaw_radian).reshape(-1)], axis=-1).reshape(1, -1)
        if (self.rotate_angle != 0):
            boxes[:, 0:3] = dust_util.rotate_lidar_along_z(boxes[:, 0:3], self.rotate_angle)
            boxes[:, 6] = dust_util.rect_to_yaw(float(boxes[:, 6]), self.rotate_angle)
        return dust_util.boxes_to_corners_3d(boxes)

    def statistic_box_distribution(self, all_corners, path_suffix, draw_flag=False, torch_save=False):
        all_box_x = all_corners[:, :, 0].flatten().tolist()
        all_box_y = all_corners[:, :, 1].flatten().tolist()
        all_box_z = all_corners[:, :, 2].flatten().tolist()
        
        save_path_root =  self.statistic_log_path / path_suffix
        save_path_root.mkdir(parents=True, exist_ok=True)
        
        if (torch_save):
            import torch
            save_path = save_path_root / f'all_box_coor.pth'
            torch.save(all_box_z, save_path)
            # a = torch.load(save_path)

        if (draw_flag):
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
            
        self.logger.info(f'save box distribution to {save_path_root}')
        
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

        self.logger.info(f'total data: {len(data_list)}, valid data: {len(valid_list)}, split rate: {self.train_split_rate}, train data: {len(train_list)}, test data: {len(val_list)}')
        
        self.statistic_label_list_box_distribution(train_list, 'split_train_data_box_dist')
        self.statistic_label_list_box_distribution(val_list, 'split_val_data_box_dist')
        
        save_split_path = self.module_root_path / 'data' / 'dust' / 'split'
        save_split_path.mkdir(parents=True, exist_ok=True)
        
        train_split_path = save_split_path / 'train.txt'
        common_util.save_to_file_line_by_line(self.logger, train_list, train_split_path)
        
        val_split_path = save_split_path / 'val.txt'
        common_util.save_to_file_line_by_line(self.logger, val_list, val_split_path)
        
    def statistic_label_list_box_distribution(self, label_list, save_path_suffix, draw_flag=False, torch_save=False):
        all_corners = None
        
        for label_name in label_list:
            objs = self.dust_dataset.get_label_objects_by_path((self.dust_dataset.label_dir / label_name).with_suffix('.txt'))
            
            for obj in objs:
                corners = self.get_rect_3dbox(obj)
                
                # box_y = corners[:, :, 1].flatten().tolist()
                # if any(elem > 2. for elem in box_y):
                #     self.logger.error(f'label: {label_name}')
                    
                if all_corners is None:
                    all_corners = corners
                else:
                    all_corners = np.concatenate((all_corners, corners), axis=1)
                    
        self.statistic_box_distribution(all_corners, save_path_suffix, torch_save=True)
    
    def statistic_all_box_distribution(self):
        self.logger.info(f'start count all box distribution: {self.raw_dataset_path}')
        
        file_list = os.listdir(self.dust_dataset.lidar_dir)
        bin_data_list = list(filter(lambda x: x.endswith('.bin'), file_list))
        data_list = [s.replace('.bin', '') for s in bin_data_list]
        invalid_label = self.get_invalid_label()
        valid_list = common_util.delete_list_elem_obtain_other(data_list, invalid_label)
        
        if len(data_list) == 0:
            self.logger.error('no bin files')
            return
        
        self.statistic_label_list_box_distribution(valid_list, 'data_box_dist', torch_save=True)
        
    def draw_3d_bbox_in_pcd_image(self, args):
        self.logger.info(f'start draw 3d bbox in pcd image: {self.raw_dataset_path}')

        raw_data_root_path = self.module_root_path.parent / 'annos' / 'raw'
        origin_data_root_path = self.module_root_path.parent / 'annos' / 'origin'
        simfusion_data_root_path = self.module_root_path.parent / 'annos' / 'simfusion'

        import torch
        data_list = os.listdir(raw_data_root_path)
        filter_id = ['1661512352_405903']
        
        for data_name in data_list:
            raw_data_path = raw_data_root_path / data_name
            origin_data_path = origin_data_root_path / data_name
            simfusion_data_path = simfusion_data_root_path / data_name
            
            raw_data = torch.load(raw_data_path)
            origin_data = torch.load(origin_data_path)
            simfusion_data = torch.load(simfusion_data_path)
        
            data_id = data_name.replace('.pth', '')
            if (data_id in filter_id):
                continue

            bbox = {}
            bbox['id'] = data_id
            bbox['boxes'] = [raw_data, origin_data, simfusion_data]
            self.dust_dataset.draw_3d_bbox_in_rect_point_cloud_image_by_index(
                rotate_angle=args.pcd_rect_rotate_angle,
                x_range=(5, 53.8),
                y_range=(-6.4, 0.8),
                z_range=(-5, 0),
                boxes=bbox
            )
            break
