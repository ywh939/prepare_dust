import numpy as np

class Object3d(object):
    ''' 3d object label '''
    def __init__(self, label_file_line) -> None:
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        self.type = data[0] # 'Car', 'Pedestrian', ...
        
        # extract 2d bounding box in image coordinates.
        self.xmin = data[4]
        self.ymin = data[5]
        self.xmax = data[6]
        self.ymax = data[7]
        self.box2d = np.hstack((self.xmin, self.ymin, self.xmax, self.ymax))
        
        # extract 3d bounding box information
        self.h = data[8] # box height
        self.w = data[9] # box width
        self.l = data[10] # box length (in meters)
        self.lwh = np.hstack((self.l, self.w, self.h))
        
        # center location (x,y,z) in bottom surface of 3d box in lidar coordiante
        self.center_bottom = np.hstack((data[11], data[12], data[13]))
        self.center = np.hstack((self.center_bottom[0:2], (self.center_bottom[...,2] + self.h / 2)))
        
        # extract yaw angle (around Y-axis in camera coordinates or around Z-axis in lidar coordinates)
        self.yaw_radian = data[14]
        
    def is_invalid(self):
        return np.any(self.lwh == 0) or np.any(self.box2d == 0)