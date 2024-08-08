from . import mine_calibration, mine_object
import os
import numpy as np
from dust.utils import dust_util, common_util
from pathlib import Path

class MineDataset(object):
    def __init__(self, logger, args):

        self.logger = logger
        self.root_path = Path(args.raw_dataset_path)
        self.label_dir = self.root_path / 'label'
        assert self.label_dir.exists()
        self.calib_dir = self.root_path / 'licam_calib'
        assert self.calib_dir.exists()
        self.lidar_dir = self.root_path / 'lidar'
        assert self.lidar_dir.exists()
        self.img_dir = self.root_path / 'left_cam'
        # self.img_dir = self.root_path / 'right_cam'
        assert self.img_dir.exists()
    
    def get_label(self, file_name):
        label_file = self.label_dir / file_name
        assert label_file.exists()
        return mine_object.get_objects_from_label(label_file)
    
    def get_calib(self, file_name):
        calib_file = self.calib_dir / file_name
        assert calib_file.exists()
        return mine_calibration.Calibration(calib_file)
    
    def get_lidar(self, file_name):
        lidar_file = self.lidar_dir / file_name
        assert lidar_file.exists()
        return dust_util.load_lidar_data(lidar_file)
    
    def get_image_by_label(self, lable_name):
        img_file = self.img_dir / lable_name
        img_file = img_file.with_suffix('.png')
        assert img_file.exists()
        return dust_util.load_image_data(img_file)

    def get_lidar_by_label(self, lable_name):
        lidar_file = self.lidar_dir / lable_name
        lidar_file = lidar_file.with_suffix('.pcd')
        assert lidar_file.exists()
        return dust_util.load_lidar_data(lidar_file)
    
    def visualize_3d_bbox_center_in_image(self, object3d, img, calib, paint_color=True, save_image=False):
        pcd_image = calib.project_point_cloud_to_image(object3d.loc.reshape(-1, 3))
        dust_util.draw_point_in_image(
            img=img, 
            pcd_image=pcd_image,
            circle_radius=5,
            save_path=Path(self.dataset_dir, f'point_in_image.png') if save_image else None
        )
    
    def visualize_3d_bbox_in_image(self, object3d, img, calib, save_image=False):
        line_coord, point_coord = dust_util.get_3d_bbox_coordinates_in_lidar(object3d.loc, object3d.lwh, object3d.rz)
        pcd_image = calib.project_point_cloud_to_image(point_coord)
        box_img = dust_util.draw_projected_bbox3d(img, pcd_image)
        dust_util.show_and_save_image(
            img=box_img,
            save_path=Path(self.root_path, f'3d_bbox.png') if save_image else None
        )
    
    def draw_3d_bbox_in_point_cloud_by_index(self, lidar, object3d, paint_color=False):
        dust_util.visualize_3D_bbox_point_cloud(
            pcd=lidar,
            center=object3d.loc,
            lwh=object3d.lwh,
            yaw=object3d.rz,
            paint_color=paint_color
        )

    def visualize_point_cloud_in_image(self, img, lidar, calib, save_image=False):
        pcd_image = calib.project_point_cloud_to_image(lidar)
        
        dust_util.draw_point_in_image(
            img=img, 
            pcd_image=pcd_image,
            save_path=Path(self.dataset_dir, f'filter_point_cloud.png') if save_image else None
        )

    def process_raw_dataset(self, args):
        
        self.count_labels()
        # self.convert_kitti()
        # self.set_split_datasest()

    def count_labels(self):
        label_list = os.listdir(self.label_dir)
        
        select_type = set([1, 2, 3])
        # select_type = set([2])
        select_sample = []
        count_type_dict = {}
        all_corners = None
        sum_z = 0
        sum_l = 0
        sum_w = 0
        sum_h = 0
        select_cnt = 0

        for label_name in label_list:
            objs = self.get_label(label_name)

            # lidar = self.get_lidar_by_label(label_name)
            # img = self.get_image_by_label(label_name)
            # calib = self.get_calib(label_name)

            # dust_util.visualize_point_cloud(lidar)
            # continue
            # self.visualize_point_cloud_in_image(img, lidar, calib)

            has_select_type = False
            for obj in objs:

                # self.draw_3d_bbox_in_point_cloud_by_index(lidar, obj)
                # self.visualize_3d_bbox_center_in_image(objs, img, calib)
                # self.visualize_3d_bbox_in_image(objs, img, calib)

                if (obj.cls_type in count_type_dict):
                    count_type_dict[obj.cls_type] += 1
                else:
                    count_type_dict[obj.cls_type] = 1

                if obj.cls_id in select_type:
                    has_select_type = True

                    boxes = np.concatenate([obj.loc, obj.lwh, np.asanyarray(obj.rz).reshape(-1)], axis=-1).reshape(1, -1)
                    corners = dust_util.boxes_to_corners_3d(boxes)
                    # corners = corners[:, :4]
                    if all_corners is None:
                        all_corners = corners
                    else:
                        all_corners = np.concatenate((all_corners, corners), axis=1)

                    sum_z += obj.loc[2]
                    sum_l += obj.l
                    sum_w += obj.w
                    sum_h += obj.h
                    select_cnt += 1
                                
            if has_select_type:
                select_sample.append(label_name)

        xl = all_corners[:, :, 0].flatten().tolist()
        yl = all_corners[:, :, 1].flatten().tolist()
        zl = all_corners[:, :, 2].flatten().tolist()
        print(f"max x: {max(xl)}, max x: {min(xl)};\
              max y: {max(yl)}, max y: {min(yl)};\
              max z: {max(zl)}, max z: {min(zl)}")

        print(count_type_dict)
        # print(len(select_sample))
        print(sum_z / select_cnt, sum_l / select_cnt, sum_w / select_cnt, sum_h / select_cnt, sum_z / select_cnt - sum_h / select_cnt / 2)

        # with open('select_sample.txt', 'w') as file: 
        #     for item in select_sample:
        #         file.write(f'{item}\n')

    def convert_kitti(self):
        self.logger.info(f"the path of datasets to convert pcd to bin: {self.root_path}")
        
        file_list = os.listdir(self.lidar_dir)
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
            
            lidar_filename = self.lidar_dir / f'{file_name}.pcd'
            dust_util.convert_pcd_to_bin(lidar_filename)
            converted_count += 1
        
        self.logger.info(f'exist {exist_bin_file_count} bin files, converted {converted_count} files')

    def set_split_datasest(self):
        self.logger.info(f'start split datasets: {self.root_path}')
        
        with open(self.root_path / 'select_train_sample.txt', 'r') as f:
            file_list = [x.strip() for x in f.readlines()]

        data_list = [s.replace('.txt', '') for s in file_list]
        
        from sklearn.model_selection import train_test_split
        train_list, val_list = train_test_split(data_list, train_size=0.7, random_state=42)

        self.logger.info(f'total data: {len(data_list)}, valid data: {len(data_list)}, split rate: {0.7}, train data: {len(train_list)}, test data: {len(val_list)}')
        
        save_split_path = self.root_path
        
        train_split_path = save_split_path / 'train.txt'
        common_util.save_to_file_line_by_line(self.logger, train_list, train_split_path)
        
        val_split_path = save_split_path / 'val.txt'
        common_util.save_to_file_line_by_line(self.logger, val_list, val_split_path)