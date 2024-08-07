import requests, os
from urllib.parse import urljoin
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import subprocess


def get_http_json_handler(url):
    # 发送 GET 请求下载文件
    response = requests.get(url)

    # 检查响应状态码
    if response.status_code == 200:
        return response.json()
    
    return None

def join_url_list(url_base, url_list):
    full_url = url_base
    for path in url_list:
        if not full_url.endswith('/'):
            full_url += '/'
        if path.startswith('/'):
            path = path[1:]
        full_url = urljoin(full_url, path)
    return full_url

def get_http_content_handler(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    return None

def get_http_content_robust_handler(url, logger):
    # 配置重试策略
    retry_strategy = Retry(
        total=5,  # 总重试次数
        backoff_factor=1,  # 每次重试的延迟时间
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]  # 设置允许重试的方法
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)

    with requests.Session() as session:
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        try:
            response = session.get(url, stream=True, timeout=(10, 30))
            response.raise_for_status()
            
            file_name = url.split('/')[-1]
            with open(file_name, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # 确保chunk不为空
                        f.write(chunk)

            with open(file_name, 'rb') as f:
                file_content = f.read()

            # 删除文件
            os.remove(file_name)

            return file_content
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch content from {url}: {e}")
            # 删除文件
            if os.path.exists(file_name):
                os.remove(file_name)
            return None
        except OSError as e:
            logger.error(f"Failed to delete the file {file_name}: {e}")
            # 删除文件
            if os.path.exists(file_name):
                os.remove(file_name)
            return None

def wget_handler(url, output):
    try:
        # 构建 wget 命令
        command = ['wget', '-O', output, url]

        # 执行命令
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        # 输出 wget 命令的执行结果（可选）
        print(result.stdout)

        # 返回 True 表示下载成功
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        print(e.stderr)
        return False