from utils import http_manager
import time, os


def is_file_empty(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            if content:
                return False  # 文件内容不为空
            else:
                return True  # 文件内容为空
    except FileNotFoundError:
        print("文件未找到")
        return True  # 如果文件未找到，也可以视为“空”

def normalize_pcd_format_from_editor(logger):
    editor_url = "http://10.0.119.87:13002"
    file_list_path = "/api/listing"
    raw_pcd_path = "/api/pcdfile"

    file_list_url = http_manager.join_url_list(editor_url, [file_list_path])
    file_list = http_manager.get_http_json_handler(file_list_url)
    if (file_list is None):
        return
    
    select_folder = set(["/tai_zhong/kuangka_pcd"])
    for file_obj in file_list:
        if file_obj['folder'] not in select_folder:
            continue
        
        file_url = http_manager.join_url_list(editor_url, [raw_pcd_path, file_obj['folder'], file_obj['file']])
        # pcd = http_manager.get_http_content_robust_handler(file_url, logger)
        http_manager.wget_handler(file_url, "output/" + file_obj['file'])
        logger.info({file_obj['file']})
        if is_file_empty("output/" + file_obj['file']):
            os.remove("output/" + file_obj['file'])
            logger.error(f"remove file {file_obj['file']}")
        time.sleep(3)
        # if pcd is None:
        #     continue

        # save_file_path = "output/" + file_obj['file']
        # with open(save_file_path, 'wb') as f:
        #     f.write(pcd)

if __name__=="__main__":
    normalize_pcd_format_from_editor()