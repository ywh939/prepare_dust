from pathlib import Path

def rename_file(file_path, start_with_char):
    # 创建 Path 对象
    path = Path(file_path)
    
    # 获取文件名和扩展名
    file_name = path.name
    base_name, ext = file_name.split('.', 1) if '.' in file_name else (file_name, '')
    
    # 确保文件名非空并且长度大于0
    if len(base_name) > 0:
        # 替换第一个字符为 start_with_char
        new_base_name = start_with_char + base_name[1:]
        # 创建新的文件名
        new_file_name = new_base_name + '.' + ext if ext else new_base_name
        new_file_path = path.with_name(new_file_name)
        
        # 重命名文件
        try:
            path.rename(new_file_path)
            print(f'File renamed to: {new_file_path}')
        except BaseException as e:
            print(e)
    else:
        print('File name is too short to rename.')

if __name__ == '__main__':
    # 示例用法
    file_path = 'D:\\detection_v2.0\\detection\\testing\\lidar\\000000.pcd'  # 替换为实际文件路径
    rename_file(file_path, '1')
