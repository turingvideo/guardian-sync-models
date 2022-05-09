import os
import json
import re
import time
import logging
from nextcloud import NextCloud
from django.conf import settings

from .models import NextCloudObject


_nxc = None
_name_path_map = []
logger = logging.getLogger(__name__)


def _get_nextcloud_client():
    global _nxc
    for i in range(3):
        if i != 0:
            time.sleep(5)
        if _nxc is None:
            try:
                _nxc = NextCloud(endpoint=os.getenv('NEXTCLOUD_URL'), user=os.getenv('NEXTCLOUD_USERNAME'), password=os.getenv('NEXTCLOUD_PASSWORD'), json_output=True)
            except Exception as e:
                logger.error('connect {0} error {1}'.format(os.getenv('NEXTCLOUD_URL'), str(e)))
        else:
            break
    return _nxc


def _get_name_path_map():
    global _name_path_map
    if _name_path_map:
        return _name_path_map
    else:
        file_path = os.path.join(settings.BASE_DIR, 'name_path_map.json')
        with open(file_path, 'r') as f:
            name_path_map = json.load(f)
        return name_path_map


def get_user_folders_list(path=None):
    # 获取用户的文件夹信息
    nxc = _get_nextcloud_client()
    r = nxc.list_folders(os.getenv('NEXTCLOUD_USERNAME'), path=path)
    if r.is_ok:
        return [NextCloudObject(**d) for d in r.data]
    else:
        return []


# 上传文件
def upload_file(local_filepath, project_name, model_tag):
    if project_name != 'guardian':
        raise Exception('Just only upload guardian projetct model file')
    model_path = _get_model_path(model_tag)
    nxc = _get_nextcloud_client()
    upload_flag = False
    try:
        r = nxc.upload_file(os.getenv('NEXTCLOUD_USERNAME'), local_filepath, model_path)
        upload_flag = r.is_ok
    except Exception as e:
        logger.error(f'Upload {project_name} projetct model {model_tag} error {str(e)}')
    return upload_flag


def _get_model_path(model_tag):
    model_name = ''
    model_version = ''
    if re.match(r'v\d+', model_tag.split('_')[-1]):
        model_name = '_'.join(model_tag.split('_')[:-1])
        model_version = model_tag.split('_')[-1]
    else:
        model_name = model_tag
        model_version = 'v0'
    print(f'model_name: {model_name}, model_version: {model_version}')
    name_path_map = _get_name_path_map()
    model_path = ''
    for model in name_path_map:
        if model['MODEL_NAME'] == model_name and model['MODEL_VERSION'] == model_version:
            model_path = model['MODEL_PATH']
            break
    return model_path


# 下载文件
def download_file(project_name, model_tag):
    if project_name != 'guardian':
        raise Exception('Just only download guardian projetct model file')
    model_path = _get_model_path(model_tag)
    basename = os.path.basename(model_path)
    full_path = os.path.join(settings.BASE_DIR, basename)
    if os.path.splitext(basename)[-1] == '.py':
        raise Exception(f'Not download python file: {basename}')
    if os.path.exists(full_path):
        if os.path.isfile(full_path):
            os.remove(full_path)
        else:
            raise Exception(f'download model_tag error: {model_tag}')
    nxc = _get_nextcloud_client()
    try:
        nxc.download_file(os.getenv('NEXTCLOUD_USERNAME'), model_path)
    except Exception as e:
        logger.error(f'{project_name} project {model_tag} model not exists, download error {str(e)}')
    return full_path


# 删除文件
def delete_file(delete_file_path):
    nxc = _get_nextcloud_client()
    r = nxc.delete_path(os.getenv('NEXTCLOUD_USERNAME'), delete_file_path)
    return r.is_ok


# 判断是否是文件夹
def is_dir(obj: NextCloudObject):
    return obj.resource_type == 'collection'
