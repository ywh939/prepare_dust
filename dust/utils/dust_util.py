import cv2
import open3d
import numpy as np

def load_image_data(img_filename):
    return cv2.imread(str(img_filename))

def load_lidar_data(lidar_filename):
    return np.asarray(open3d.io.read_point_cloud(str(lidar_filename)).points, dtype=np.float32)

def load_lidar_bin_data(lidar_filename):
    return np.fromfile(str(lidar_filename), dtype=np.float32).reshape(-1, 3)

def convert_pcd_to_bin(lidar_filename):
    pcd = load_lidar_data(lidar_filename)
    bin_file_name = str(lidar_filename).replace('.pcd', '.bin')
    pcd.tofile(bin_file_name)

def _get_rotation_matrix_along_z(angle):
    randian = np.deg2rad(angle)
    return np.float64([
        np.cos(randian), -np.sin(randian), 0,
        np.sin(randian), np.cos(randian),  0,
        0,               0              ,  1
    ]).reshape(3, 3)
    
def rotate_lidar_along_z(lidar, angle):
    # lidar: (N, 3), angle: float
    # return: (N, 3)
    
    rotation_matrix = _get_rotation_matrix_along_z(angle)
    new_lidar = lidar @ rotation_matrix.T
    return new_lidar

def anti_rotate_lidar_along_z(lidar, angle):
    # lidar: (N, 3), angle: float
    # return: (N, 3)
    
    rotation_matrix = _get_rotation_matrix_along_z(angle)
    new_lidar = lidar @ rotation_matrix
    return new_lidar

def rect_to_yaw(yaw_radian, rotate_angle):
    return yaw_radian + np.deg2rad(rotate_angle)
   
def filter_point_cloud(pcd, x_range, y_range, z_range):
    # pcd: (N, 3), x_range: (x_min, x_max), y_range: (y_min, y_max), z_range: (z_min, z_max)
    # return: (M, 3)
    
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range
    
    # filter points by x, y
    x_pcd = pcd[:, 0]
    y_pcd = pcd[:, 1]
    x_filter = np.logical_and((x_pcd >= x_min), (x_pcd < x_max))
    y_filter = np.logical_and((y_pcd >= y_min), (y_pcd < y_max))
    xy_filter = np.logical_and(x_filter, y_filter)
    xy_filter_pcd = pcd[xy_filter]
    
    # filter points by z
    z_pcd = xy_filter_pcd[:, 2]
    z_filter = np.logical_and(z_pcd >= z_min, z_pcd < z_max)
    xyz_filter_pcd = xy_filter_pcd[z_filter]
    
    return xyz_filter_pcd

def get_point_cloud_visualizer(pcd, paint_color=False):
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)
    axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(axis_pcd)
    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(pcd)
    vis.add_geometry(pts)
    if not paint_color:
        pts.colors = open3d.utility.Vector3dVector(np.ones((pcd.shape[0], 3)))
    return vis

def _run_open3d_visualizer(visualizer):
    visualizer.run()
    visualizer.destroy_window()
    
def visualize_point_cloud(pcd, paint_color=False):
    vis = get_point_cloud_visualizer(pcd, paint_color)
    _run_open3d_visualizer(vis)
    
def get_rotate_x_matrix(radian):
    ''' 3D Rotation about the x-axis. '''
    c = np.cos(radian)
    s = np.sin(radian)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])

def get_rotate_y_matrix(radian):
    ''' Rotation about the y-axis. '''
    c = np.cos(radian)
    s = np.sin(radian)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])

def get_rotate_z_matrix(radian):
    ''' Rotation about the z-axis. '''
    c = np.cos(radian)
    s = np.sin(radian)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])
    
def get_rotate_x_matrix_by_open3d(radian):
    # yaw: float
    # return: (3, 3)
    
    return open3d.geometry.get_rotation_matrix_from_axis_angle(np.array([radian, 0, 0]))
    
def get_rotate_y_matrix_by_open3d(radian):
    # yaw: float
    # return: (3, 3)
    
    return open3d.geometry.get_rotation_matrix_from_axis_angle(np.array([0, radian, 0]))
    
def get_rotate_z_matrix_by_open3d(radian):
    # yaw: float
    # return: (3, 3)
    
    return open3d.geometry.get_rotation_matrix_from_axis_angle(np.array([0, 0, radian]))

def get_3d_bbox_coordinates_in_lidar(center, lwh, yaw):
    rot = get_rotate_z_matrix_by_open3d(yaw)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)
    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
    
    coordinates = []
    for i in range (len(line_set.lines)):
        coordinates.append(line_set.get_line_coordinate(i))
    
    numpy_coordinates = np.array(coordinates)
    reshpae_coordinates = numpy_coordinates.reshape(-1, 3)
    
    return numpy_coordinates, reshpae_coordinates
    # unique_coordinates = np.unique(np.array(coordinates).reshape(-1, 3), axis=0)
    
def _get_3d_bbox_lines_in_point_cloud(center, lwh, yaw, paint_color=False, line_color=(0, 1, 0)):
    rot = get_rotate_z_matrix_by_open3d(yaw)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)
    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
    lines = np.asarray(line_set.lines)
    # lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)
    line_set.lines = open3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color(line_color)
    return line_set
    
def visualize_3D_bbox_point_cloud(pcd, center, lwh, yaw, paint_color=False, line_color=(0, 1, 0)):
    vis = get_point_cloud_visualizer(pcd, paint_color)
    line_set = _get_3d_bbox_lines_in_point_cloud(center, lwh, yaw, paint_color, line_color)
    vis.add_geometry(line_set)
    _run_open3d_visualizer(vis)
    
def show_image(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    
def show_and_save_image(img, save_path):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.imwrite(str(save_path), img)
    
def draw_point_in_image(img, pcd_image, circle_radius=1, save_path=None):
    for i in range(pcd_image.shape[0]):
        cv2.circle(img, (int(pcd_image[i,0]), int(pcd_image[i,1])), circle_radius, (0, 255, 0), -1)
    show_image(img=img) if save_path is None else show_and_save_image(img=img, save_path=save_path)
    
def draw_projected_box3d(image, qs, color=(0, 255, 0), thickness=2):
    ''' Draw 3d bounding box in image
        qs: (8, 3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        # Ref: https://docs.opencv.org/3.0-beta/modules/imgproc/doc/drawing_functions.html
        i, j = k, (k+1)%4
        cv2.line(image, (qs[i,0], qs[i,1]), (qs[j,0], qs[j,1]), color, thickness, cv2.LINE_AA)
        i, j = k+4, (k+1)%4 + 4
        cv2.line(image, (qs[i,0], qs[i,1]), (qs[j,0], qs[j,1]), color, thickness, cv2.LINE_AA)
        i, j = k, k+4
        cv2.line(image, (qs[i,0], qs[i,1]), (qs[j,0], qs[j,1]), color, thickness, cv2.LINE_AA)
    return image

def draw_projected_bbox3d(img, point_coord, color=(0, 255, 0), thickness=2):
    line_coord = point_coord.reshape(12, 2, 2).astype(np.int32)
    for i in range(len(line_coord)):
        # pt1 = (line_coord[i][0][0], line_coord[i][0][1])
        # pt2 = (line_coord[i][1][0], line_coord[i][1][1])
        pt1 = tuple(line_coord[i][0])
        pt2 = tuple(line_coord[i][1])
        cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)
    return img

def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:
    """
    cosa = np.cos(angle)
    sina = np.sin(angle)
    zeros = np.zeros(points.shape[0])
    ones = np.ones(points.shape[0])
    rot_matrix = np.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), axis=1).reshape(-1, 3, 3)
    points_rot = np.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = np.concatenate((points_rot, points[:, :, 3:]), axis=-1)
    return points_rot

def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    template = np.array((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d_rel = np.tile(boxes3d[:, None, 3:6], (1, 8, 1)) * template[None, :, :]
    
    corners3d = rotate_points_along_z(corners3d_rel.reshape(-1, 8, 3), boxes3d[:, 6]).reshape(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]
    
    # line_coord, point_coord = get_3d_bbox_coordinates_in_lidar(
    #     boxes3d[:, 0:3].reshape(-1), boxes3d[:, 3:6].reshape(-1), float(boxes3d[:, 6].reshape(-1)))
    
    return corners3d

def mask_boxes_outside_range_numpy(boxes, limit_range, rotate_angle, min_num_corners=1):
    """
    Args:
        boxes: (N, 7) [x, y, z, dx, dy, dz, heading, ...], (x, y, z) is the box center
        limit_range: [minx, miny, minz, maxx, maxy, maxz]
        min_num_corners:

    Returns:

    """
    # ret = rotate_points_along_z(boxes[None, :, 0:3], np.deg2rad(rotate_angle).reshape(-1))
    boxes[:, 0:3] = rotate_lidar_along_z(boxes[:, 0:3], rotate_angle)
    boxes[:, 6] = rect_to_yaw(float(boxes[:, 6]), rotate_angle)
    
    corners = boxes_to_corners_3d(boxes)  # (N, 8, 3)
    
    mask = ((corners >= limit_range[0:3]) & (corners <= limit_range[3:6]))
    # mask = ((corners >= limit_range[0:3]) & (corners <= limit_range[3:6])).all(axis=2)

    corner_num = mask.all(axis=2).sum(axis=1)
    
    return mask, corner_num, corners

def draw_hist(data, title, xlabel, ylabel, save_path=None):
    import matplotlib.pyplot as plt
    bins=int(len(data)/5)
    plt.hist(data, bins=bins)
    # plt.hist(data, density=False, bins=bins, align='left')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()