from pathlib import Path
import numpy as np
import os

from dust.utils import dust_util as dust_util
from dust import dust_label

class Calibration(object):
    def __init__(self) -> None:
        
        self.lidar2camera_rotation_matrix = np.float32([
            -0.87379163, -0.48617069, 0.01123575,
            -0.00201444, -0.01948573, -0.99980811,
            0.48629633,  -0.87364659, 0.0160471,
        ]).reshape(3,3)
        
        self.lidar2camera_transition_vector = np.float32([
            -0.0032562, -0.1556693, -0.21232637
        ]).reshape(3,1)
        
        self.camera_intrinsic_matrix = np.float32([
            862.7151336, 0,            472.99592418,
            0,           862.56913021, 314.58265642,
            0,           0,            1,
        ]).reshape(3,3)
        
    def project_point_cloud_to_camera(self, pcd):
        # pcd: (N, 3)
        # return: (N, 3)
        
        pcd_camera = pcd @ self.lidar2camera_rotation_matrix.T + self.lidar2camera_transition_vector.T
        
        v2c = np.hstack((self.lidar2camera_rotation_matrix, self.lidar2camera_transition_vector))
        lidar2cam_quantic = np.vstack((v2c, np.array([0, 0, 0, 1], dtype=np.float32).reshape(1, 4)))
        pcd_camera2 = np.matmul(np.hstack((pcd, np.ones((pcd.shape[0], 1), dtype=np.float32))), lidar2cam_quantic.T)
        
        return pcd_camera
    
    def rect_cameara_to_image(self, pcd_camera):
        # pcd_camera: (N, 3)
        # return: (N, 2)
        
        pcd_rect = pcd_camera @ self.camera_intrinsic_matrix.T
        pcd_image = pcd_rect[:, 0:2] / pcd_rect[:, 2:]
        return pcd_image
        
    def project_point_cloud_to_image(self, pcd):
        # pcd: (N, 3)
        # return: (N, 2)
        pcd_camera = self.project_point_cloud_to_camera(pcd)
        pcd_image = self.rect_cameara_to_image(pcd_camera)
        return pcd_image
        
    def project_rect_point_cloud_to_image(self, pcd):
        # pcd: (N, 3)
        # return: (N, 2)
        pcd_camera = self.project_point_cloud_to_camera(pcd)
        pcd_image = self.rect_cameara_to_image(pcd_camera)
        return pcd_image

class DustDataset(object):
    def __init__(self, dataset_dir) -> None:
        self.dataset_dir = dataset_dir
        
        self.image_dir = self.dataset_dir / 'image_camera'
        self.lidar_dir = self.dataset_dir / 'pcd'
        self.label_dir = self.dataset_dir / 'annotation' / 'kitti_imagecamera'
        
        self.calibration = Calibration()
        
    def get_image_by_idx(self, idx):
        image_filename = Path(self.image_dir, f'{idx}.png')
        return dust_util.load_image_data(image_filename)
    
    def get_lidar_by_idx(self, idx):
        lidar_filename = Path(self.lidar_dir, f'{idx}.pcd')
        return dust_util.load_lidar_data(lidar_filename)
    
    def get_lidar_bin_by_idx(self, idx):
        lidar_filename = Path(self.lidar_dir, f'{idx}.bin')
        return dust_util.load_lidar_bin_data(lidar_filename)
    
    def get_label_objects(self, idx):
        label_filename = Path(self.label_dir, f'{idx}.txt')
        with open(label_filename, 'r') as f:
            lines = f.readlines()
        return [dust_label.Object3d(line) for line in lines]
    
    def get_label_objects_by_path(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        return [dust_label.Object3d(line) for line in lines]
    
    def convert_pcd_to_bin(self, idx):
        lidar_filename = Path(self.lidar_dir, f'{idx}.pcd')
        dust_util.convert_pcd_to_bin(lidar_filename)
        
    def get_anchor_size(self, label_path):
        import os
        
        label_list = os.listdir(label_path)
        
        car_size = np.zeros(3)
        car_count = 0
        
        for label_name in label_list:
            objs = self.get_label_objects_by_path(label_path / label_name)
            for obj in objs:
                if obj.type == 'Car':
                    if (obj.lwh <= 0).all():
                        print(f"label {label_name}, object type {obj.type}, Invalid size: {obj.lwh}")
                        continue
                    
                    car_size += obj.lwh
                    car_count += 1
                else:
                    print(f'label {label_name}, Not Car: {obj.type}')
        
        anchor_size = car_size / car_count
        print(f'anchor size: {anchor_size}')
        
        return anchor_size
                    
    def check_box_outside_range(self, logger, label_path, limit_range, rotate_angle):
        from collections import defaultdict
        
        label_list = os.listdir(label_path)
        min_z = limit_range[2]
        min_z_label = ''
        corner_cnt = np.zeros(8).reshape(8, 1)
        corner_name_cnt = defaultdict(list)
        check_label = []
        all_invalid_corners = None
        all_corners = None
        
        for label_name in label_list:
            objs = self.get_label_objects_by_path(label_path / label_name)
            objs_num = len(objs)
            if objs_num == 0:
                logger.error(f'invalid objects num: {objs_num}')
                continue
            
            for obj in objs:
                if obj.type != 'Car':
                    logger.error(f'label {label_name}, Not Car: {obj.type}')
                    continue
                
                if obj.type == 'Car':
                    if (obj.lwh <= 0).all():
                        # print(f"label {label_name}, object type {obj.type}, Invalid size: {obj.lwh}")
                        continue
                    
                    boxes = np.concatenate([obj.center, obj.lwh, np.asanyarray(obj.yaw_radian).reshape(-1)], axis=-1)
                    mask, corner_num, corners = dust_util.mask_boxes_outside_range_numpy(boxes.reshape(1, -1), limit_range, rotate_angle)
                    if all_corners is None:
                        all_corners = corners
                    else:
                        all_corners = np.concatenate((all_corners, corners), axis=1)
                    
                    if label_name.split('.')[0] in check_label:
                        pass
                        # idx = np.argwhere(mask==False)
                        # # val = corners[~mask].reshape(8, -1)
                        # val = corners[idx[..., :, 0], idx[..., :, 1], idx[..., :, 2]]
                        # inval_corners = np.hstack((idx, val))
                    
                    if corner_num < 8:
                        # print(f'label {label_name}, corner num {corner_num}')
                        corner_cnt[corner_num] += 1
                        corner_name_cnt[int(corner_num)].append(label_name)
                        idx = np.argwhere(mask==False)
                        val_m = corners[~mask].reshape(idx.shape[0], -1)
                        val = corners[idx[..., :, 0], idx[..., :, 1], idx[..., :, 2]].reshape(idx.shape[0], -1)
                        inval_corners = np.hstack((idx, val))
                        # print(f'label {label_name}, corner num {corner_num}, invalid corner: {inval_corners[:, 2:4]}')
                        if all_invalid_corners is None:
                            all_invalid_corners = inval_corners
                        else:
                            all_invalid_corners = np.vstack((all_invalid_corners, inval_corners))
                        
                        axis = inval_corners[:, 2].flatten().tolist()
                        if 0.0 in axis:
                            print(f'label {label_name}, corner num {corner_num}, invalid x corner: {inval_corners[:, 2:4]}')
                        elif 1.0 in axis:
                            print(f'label {label_name}, corner num {corner_num}, invalid y corner: {inval_corners[:, 2:4]}')
                        elif 2.0 in axis:
                            print(f'label {label_name}, corner num {corner_num}, invalid z corner: {inval_corners[:, 2:4]}')
                        
                    if corner_num == 0:
                        continue
                            
                    if min_z > np.min(corners, axis=1)[0, 2]:
                        min_z = np.min(corners, axis=1)[0, 2]
                        min_z_label = label_name
                        # print(f'label {label_name}, min_z {min_z}')

        if all_invalid_corners is not None:
            result_dict = {}
            for row in all_invalid_corners:
                key = row[2]
                val = row[3]
                if key not in result_dict:
                    result_dict[key] = []
                result_dict[key].append(val)
        
        # for key in result_dict.keys():
        #     dust_util.draw_hist(data=result_dict[key],
        #                         title=f'lidar box corner distribution',
        #                         xlabel=f'{xyz_name[int(key)]} coordinate',
        #                         ylabel='count'
        #                         )
            
        # all_box_x = all_corners[:, :, 0].flatten().tolist()
        # all_box_y = all_corners[:, :, 1].flatten().tolist()
        # all_box_z = all_corners[:, :, 2].flatten().tolist()
        # save_path_root = Path(r'C:\Users\Y\Documents')
        # dust_util.draw_hist(data=all_box_x,
        #                     title=f'lidar box x coordinate distribution',
        #                     xlabel=f'x coordinate value',
        #                     ylabel='count',
        #                     save_path=save_path_root / f'x_coordinate_distribution.png'
        #                     )
        # dust_util.draw_hist(data=all_box_y,
        #                     title=f'lidar box y coordinate distribution',
        #                     xlabel=f'y coordinate value',
        #                     ylabel='count',
        #                     save_path=save_path_root / f'y_coordinate_distribution.png'
        #                     )
        # dust_util.draw_hist(data=all_box_z,
        #                     title=f'lidar box z coordinate distribution',
        #                     xlabel=f'z coordinate value',
        #                     ylabel='count',
        #                     save_path=save_path_root / f'z_coordinate_distribution.png'
        #                     )
        
        print(f'label {min_z_label} min z: {min_z}')

    def visualize_point_cloud_by_idx(self, idx, paint_color=False):
        lidar = self.get_lidar_by_idx(idx)
        dust_util.visualize_point_cloud(lidar, paint_color)
        
    def visualize_point_cloud_by_idx_and_angle(self, idx, angle, paint_color=False):
        lidar = self.get_lidar_by_idx(idx)
        rotated_lidar = dust_util.rotate_lidar_along_z(lidar, angle)
        dust_util.visualize_point_cloud(rotated_lidar, paint_color)
        
    def visualize_point_cloud_by_idx_and_filter(self, idx, x_range, y_range, z_range, paint_color=False):
        lidar = self.get_lidar_by_idx(idx)
        filtered_lidar = dust_util.filter_point_cloud(lidar, x_range, y_range, z_range)
        dust_util.visualize_point_cloud(filtered_lidar, paint_color)
        
    def visualize_point_cloud_by_idx_and_angle_and_filter(self, idx, rotate_angle, x_range, y_range, z_range, paint_color=False):
        lidar = self.get_lidar_by_idx(idx)
        rotated_lidar = dust_util.rotate_lidar_along_z(lidar, rotate_angle)
        filtered_lidar = dust_util.filter_point_cloud(rotated_lidar, x_range, y_range, z_range)
        dust_util.visualize_point_cloud(filtered_lidar, paint_color)
        
    def draw_3d_bbox_in_rect_point_cloud_by_index(self, idx, rotate_angle, x_range, y_range, z_range, paint_color=False, line_color=(0, 1, 0)):
        lidar = self.get_lidar_by_idx(idx)
        rotated_lidar = dust_util.rotate_lidar_along_z(lidar, rotate_angle)
        filtered_lidar = dust_util.filter_point_cloud(rotated_lidar, x_range, y_range, z_range)
        object3d = self.get_label_objects(idx)
        
        rect_center = dust_util.rotate_lidar_along_z(object3d[0].center, rotate_angle)
        
        rect_cen_bo = dust_util.rotate_lidar_along_z(object3d[0].center_bottom, rotate_angle)
        # center = np.hstack((rect_cen_bo[0:2], (rect_cen_bo[...,2] + object3d[0].h / 2)))
        
        rect_yaw = dust_util.rect_to_yaw(object3d[0].yaw_radian, rotate_angle)
        
        dust_util.visualize_3D_bbox_point_cloud(
            pcd=filtered_lidar,
            center=rect_center,
            lwh=object3d[0].lwh,
            yaw=rect_yaw,
            paint_color=paint_color
        )
    
    def draw_3d_bbox_in_rect_point_cloud_image_by_index(self, rotate_angle, x_range, y_range, z_range, boxes, paint_color=False):
        lidar_id = boxes['id']

        lidar = self.get_lidar_by_idx(lidar_id)
        rotated_lidar = dust_util.rotate_lidar_along_z(lidar, rotate_angle)
        filtered_lidar = dust_util.filter_point_cloud(rotated_lidar, x_range, y_range, z_range)
        save_path = Path(str(self.dataset_dir / 'box_img') + '/' + lidar_id + '.png')

        dust_util.visualize_3D_bbox_point_cloud_in_image(
            save_path=save_path,
            pcd=filtered_lidar,
            boxes= boxes['boxes'],
            paint_color=paint_color
        )
        
    def draw_3d_bbox_in_filter_point_cloud_by_index(self, idx, x_range, y_range, z_range, paint_color=False, line_color=(0, 1, 0)):
        lidar = self.get_lidar_by_idx(idx)
        filtered_lidar = dust_util.filter_point_cloud(lidar, x_range, y_range, z_range)
        object3d = self.get_label_objects(idx)
        
        dust_util.visualize_3D_bbox_point_cloud(
            pcd=filtered_lidar,
            center=object3d[0].center,
            lwh=object3d[0].lwh,
            yaw=object3d[0].yaw_radian,
        )
        
    def visualize_filter_point_cloud_in_fov(self, idx, x_range, y_range, z_range, paint_color=False):
        lidar = self.get_lidar_by_idx(idx)
        filtered_lidar = dust_util.filter_point_cloud(lidar, x_range, y_range, z_range)
        pcd_image = self.calibration.project_point_cloud_to_image(filtered_lidar)
        
        img = self.get_image_by_idx(idx)
        img_height, img_width = img.shape[0:2]
        xmin = ymin = 0,
        xmax, ymax = img_width, img_height
        
        fov_inds = (pcd_image[:,0]<xmax) & (pcd_image[:,0]>=xmin) & \
                 (pcd_image[:,1]<ymax) & (pcd_image[:,1]>=ymin)
        imgfov_poins = filtered_lidar[fov_inds, :]
        
        dust_util.visualize_point_cloud(pcd=imgfov_poins)
        
    def visualize_3d_bbox_center_in_image(self, idx, paint_color=True, save_image=False):
        object3d = self.get_label_objects(idx)
        pcd_image = self.calibration.project_point_cloud_to_image(object3d[0].center)
        dust_util.draw_point_in_image(
            img=self.get_image_by_idx(idx), 
            pcd_image=pcd_image,
            circle_radius=5,
            save_path=Path(self.dataset_dir, f'{idx}_point_in_image.png') if save_image else None
        )
        
    def visualize_3d_bbox_in_image(self, idx, paint_color=True, save_image=False):
        object3d = self.get_label_objects(idx)
        
        line_coord, point_coord = dust_util.get_3d_bbox_coordinates_in_lidar(object3d[0].center, object3d[0].lwh, object3d[0].yaw_radian)
        pcd_image = self.calibration.project_point_cloud_to_image(point_coord)
        img = dust_util.draw_projected_bbox3d(self.get_image_by_idx(idx), pcd_image)
        dust_util.show_and_save_image(
            img=img,
            save_path=Path(self.dataset_dir, f'{idx}_3d_bbox.png') if save_image else None
        )
        
        # l, w, h = object3d[0].l, object3d[0].w, object3d[0].h
        
        # # 3d bounding box corners, should be consistent with draw_projected_box3d
        # x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        # y_corners = [-w/2, w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2]
        # z_corners = [h, h, h, h, 0, 0, 0, 0]
        
        # # yaw matrix
        # yaw_radian = object3d[0].yaw_radian
        # yaw_matrix = dust_util.get_rotate_z_matrix(yaw_radian)
        
        # # rotate and translate 3d bounding box
        # corners_3d = np.vstack([x_corners, y_corners, z_corners])
        # corners_3d = np.dot(yaw_matrix, corners_3d)
        # corners_3d += object3d[0].center_bottom.reshape((3,1))
        
        # pcd_image = self.calibration.project_point_cloud_to_image(corners_3d.T)
        
        # dust_util.show_and_save_image(
        #     img=dust_util.draw_projected_box3d(self.get_image_by_idx(idx), pcd_image),
        #     save_path=Path(self.dataset_dir, f'{idx}_3d_bbox.png') if save_image else None
        # )
        
    def visualize_filter_point_cloud_in_image(self, idx, x_range, y_range, z_range, paint_color=False, save_image=False):
        lidar = self.get_lidar_by_idx(idx)
        filtered_lidar = dust_util.filter_point_cloud(lidar, x_range, y_range, z_range)
        pcd_image = self.calibration.project_point_cloud_to_image(filtered_lidar)
        
        dust_util.draw_point_in_image(
            img=self.get_image_by_idx(idx), 
            pcd_image=pcd_image,
            save_path=Path(self.dataset_dir, f'{idx}_filter_point_cloud.png') if save_image else None
        )
        
    def visualize_rotate_and_filter_point_cloud_in_image(self, idx, rotate_angle, x_range, y_range, z_range, paint_color=False, save_image=False):
        lidar = self.get_lidar_by_idx(idx)
        rotated_lidar = dust_util.rotate_lidar_along_z(lidar, rotate_angle)
        filtered_lidar = dust_util.filter_point_cloud(rotated_lidar, x_range, y_range, z_range)
        
        anti_rotate_lidar = dust_util.anti_rotate_lidar_along_z(filtered_lidar, rotate_angle)
        pcd_image = self.calibration.project_point_cloud_to_image(anti_rotate_lidar)
        
        dust_util.draw_point_in_image(
            img=self.get_image_by_idx(idx), 
            pcd_image=pcd_image,
            save_path=Path(self.dataset_dir, f'{idx}_rotate_filter_point_cloud.png') if save_image else None
        )
        
        
if __name__=='__main__':
    dust_dataset = DustDataset(Path(r'C:\Users\Y\Documents\dust_datasets\dust'))
    dust_dataset.visualize_point_cloud_by_idx('1661512340_793590')