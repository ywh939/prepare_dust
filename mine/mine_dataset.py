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

    def count_labels(self):
        label_list = os.listdir(self.label_dir)
        
        select_type = set([1, 2, 3])
        select_sample = []
        count_type_dict = {}
        for label_name in label_list:
            objs = self.get_label(label_name)

            lidar = self.get_lidar_by_label(label_name)
            img = self.get_image_by_label(label_name)
            calib = self.get_calib(label_name)

            # dust_util.visualize_point_cloud(lidar)
            # self.visualize_point_cloud_in_image(img, lidar, calib)

            has_select_type = False
            for obj in objs:
                # self.draw_3d_bbox_in_point_cloud_by_index(lidar, obj)
                # self.visualize_3d_bbox_center_in_image(objs, img, calib)
                # self.visualize_3d_bbox_in_image(objs, img, calib)

                # if (obj.cls_type in count_type_dict):
                #     count_type_dict[obj.cls_type] += 1
                # else:
                #     count_type_dict[obj.cls_type] = 1

                if obj.cls_id in select_type:
                    has_select_type = True
                    continue
            
            if has_select_type:
                select_sample.append(label_name)

        # print(count_type_dict)
        print(len(select_sample))
        with open('select_sample.txt', 'w') as file: 
            for item in select_sample:
                file.write(f'{item}\n')