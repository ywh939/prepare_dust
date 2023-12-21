import logging

def create_logger(log_file=None, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s [%(filename)s:%(lineno)d] %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

def save_to_file_line_by_line(logger, val_list, save_path):
    with open(save_path, 'w') as file:
        file.write('\n'.join(val_list))
    logger.info(f'save {len(val_list)} value to {save_path}')
    
def delete_list_elem_obtain_other(a_list, b_list):
    # 转换为集合
    a_set = set(a_list)
    b_set = set(b_list)

    # 从a_set中删除与b_set相同的元素
    result_set = a_set - b_set

    # 转换回列表
    return list(result_set)