from . import kitti_calibration, kitti_object
import os
import numpy as np
from dust.utils import dust_util
from pathlib import Path

class KittiDataset(object):
    def __init__(self, logger, args):

        self.logger = logger
        self.root_path = Path(args.raw_dataset_path)
        self.label_dir = self.root_path / 'label_2'
        assert self.label_dir.exists()
    
    def get_label(self, file_name):
        label_file = self.label_dir / file_name
        return kitti_object.get_objects_from_label(label_file)
    
    def get_calib(self, file_name):
        calib_file = self.root_path / 'calib' / file_name
        assert calib_file.exists()
        return kitti_calibration.Calibration(calib_file)

    def count_labels(self):
        label_list = os.listdir(self.label_dir)
        
        all_corners = None
        for label_name in label_list:
            objs = self.get_label(label_name)
            for obj in objs:
                if obj.cls_type != 'Car':
                    continue

                calib = self.get_calib(label_name)
                loc_lidar = calib.rect_to_lidar(obj.loc.reshape(-1, 3))
                loc_lidar[:, 2] += obj.h / 2
                boxes = np.concatenate([loc_lidar.reshape(-1), obj.lwh, np.asanyarray(obj.ry).reshape(-1)], axis=-1).reshape(1, -1)
                corners = dust_util.boxes_to_corners_3d(boxes)
                if all_corners is None:
                    all_corners = corners
                else:
                    all_corners = np.concatenate((all_corners, corners), axis=1)
                    
        all_box_z = all_corners[:, :, 2].flatten().tolist()
        
        import torch
        save_path = self.root_path / f'all_box_coor_z.pth'
        torch.save(all_box_z, save_path)