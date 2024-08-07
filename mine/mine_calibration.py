import numpy as np


def format_calib_str(calib_str):
    return np.array(calib_str.strip().split(' '), dtype=np.float32).reshape(1, 4)

def get_calib_from_file(calib_file):
    with open(calib_file) as f:
        lines = f.readlines()

    ca1 = format_calib_str(lines[0])
    ca2 = format_calib_str(lines[1])
    ca3 = format_calib_str(lines[2])
    ca4 = format_calib_str(lines[3])

    li2cam = np.concatenate((ca1, ca2, ca3, ca4), axis=0)

    return li2cam


class Calibration(object):
    def __init__(self, calib_file):
        if not isinstance(calib_file, dict):
            calib = get_calib_from_file(calib_file)
        else:
            calib = calib_file

        self.V2C = calib  # 4 x 4
        self.camera_intrinsic_matrix = np.float32([
            1484.74555368, 0,          1120.46023222,
            0,           1217.0010484, 808.94286133,
            0,           0,            1,
        ]).reshape(3,3)

    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def project_point_cloud_to_image(self, pcd):
        # pcd: (N, 3)
        # return: (N, 2)

        # pcd = np.array((11.5519214 ,  4.80974081, -1.32543733)).reshape(-1, 3) # test bottom corner
        pcd_camera = self.project_point_cloud_to_camera(pcd)
        pcd_image = self.rect_cameara_to_image(pcd_camera)
        return pcd_image
    
    def rect_cameara_to_image(self, pcd_camera):
        # pcd_camera: (N, 3)
        # return: (N, 2)
        
        pcd_rect = pcd_camera @ self.camera_intrinsic_matrix.T
        pcd_image = pcd_rect[:, 0:2] / pcd_rect[:, 2:]
        # pcd_image = np.array((680, 840), dtype=np.float32).reshape(1, 2) # test center
        # pcd_image = np.array((500, 980), dtype=np.float32).reshape(1, 2) # test bottom corner
        # print(pcd_image)
        return pcd_image

    def project_point_cloud_to_camera(self, pcd):
        # pcd: (N, 3)
        # return: (N, 3)
        
        pts_rect_hom = self.cart_to_hom(pcd)  # (N, 4)
        pts_lidar = np.dot(pts_rect_hom, self.V2C.T)
        return pts_lidar[:, 0:3]