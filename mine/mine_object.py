import numpy as np


def get_objects_from_label(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    objects = [Object3d(line) for line in lines]
    return objects

def cls_type_to_id(cls_type):
    type_to_id = {'Mining-Truck': 1, 'Wide-Body-Truck': 2, 'water_car': 3}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]


class Object3d(object):
    def __init__(self, line):
        label = line.strip().split(' ')
        
        self.cls_type = label[0]
        self.cls_id = cls_type_to_id(self.cls_type)
        self.truncation = float(label[1])
        self.occlusion = float(label[2])  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        self.box2d = np.array((float(label[3]), float(label[4]), float(label[5]), float(label[6])), dtype=np.float32)
        self.l = float(label[7])
        self.w = float(label[8])
        self.h = float(label[9])
        self.lwh = np.hstack((self.l, self.w, self.h))

        self.loc = np.array((float(label[12]), float(label[10]), float(label[11])), dtype=np.float32)
        # self.loc[2] -= self.h / 2
        
        self.ry = float(label[13])
        self.rz = float(label[14])
        self.rx = float(label[15])

    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)],
                      [0, 1, 0],
                      [-np.sin(self.ry), 0, np.cos(self.ry)]])
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.loc
        return corners3d

    def to_str(self):
        print_str = '%s %.3f %.3f %.3f box2d: %s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f' \
                     % (self.cls_type, self.truncation, self.occlusion, self.box2d, self.l, self.w, self.h,
                        self.loc, self.ry, self.rz, self.rx)
        return print_str
    
    def to_label_str(self):
        label_str = f"{self.cls_type} {self.truncation} {self.occlusion} {self.box2d[0]} "\
                    f"{self.box2d[1]} {self.box2d[2]} {self.box2d[3]} {self.l} {self.w} {self.h} "\
                    f"{self.loc[1]} {self.loc[2]} {self.loc[0]} {self.ry} {self.rz} {self.rx}"
        return label_str

    @staticmethod
    def set_objects_to_label_file(objs, label_path):
        with open(label_path, 'w') as f:
            for obj in objs:
                f.write(obj.to_label_str())
                f.write("\n")