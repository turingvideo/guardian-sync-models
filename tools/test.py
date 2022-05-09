import argparse
import requests
import hashlib
import logging
import os
import shutil
import tempfile
import time
import zipfile
from collections import OrderedDict
from model_config import ModelSettings, default_model_setting


timeout = 5
proxies = {"http": "1", "https": "http://1.1.1.8:889"}
proxies = None
project_name = 'guardian'
logger = logging.getLogger(__name__)


# 压缩文件夹
def zip_obj(zipname, obj_path):
    if not os.path.exists(obj_path):
        raise Exception(f'{obj_path} not exists, not can zip')
    if os.path.exists(zipname):
        os.remove(zipname)
    with zipfile.ZipFile(zipname, 'w') as zip:
        if os.path.isfile(obj_path):
            zip.write(obj_path, os.path.basename(obj_path))
        else:
            for path, _, filenames in os.walk(obj_path):
                fpath = path.replace(obj_path, os.path.basename(obj_path))
                for filename in filenames:
                    zip.write(os.path.join(path, filename), os.path.join(fpath, filename))
                if len(filenames) == 0:
                    zip.write(path, fpath)


# 解压文件夹
def unzip_obj(obj_path, unzip_path):
    if not os.path.exists(obj_path):
        raise Exception(f'{obj_path} not exists, not can unzip')
    if not os.path.exists(unzip_path):
        os.makedirs(unzip_path)
    if zipfile.is_zipfile(obj_path):
        with zipfile.ZipFile(obj_path, 'r') as zip:
            zip.extractall(path=unzip_path)
    else:
        shutil.move(obj_path, unzip_path)


def get_s3_url(session, project, model_name, type: str, timeout=5):
    login_url = "http://127.0.0.1:8000/overlord/user/login"
    login_body = {
        'email': 'guardian+m3@turingvideo.com',
        'password': ')S1I!2<:Uq=j'
    }

    login_response = session.post(login_url, json=login_body, timeout=timeout, proxies=proxies)
    login_response.raise_for_status()

    get_s3_url_url = \
        "http://127.0.0.1:8000/scv/s3url/model/{type}/{project}/{name}" \
        .format(type=type, project=project, name=model_name)

    response = session.get(get_s3_url_url, timeout=timeout, proxies=proxies)
    response.raise_for_status()
    result = response.json()
    print(result)
    return result


def upload_file(url: str, file_path: str, fields: dict):
    file_name = os.path.abspath(file_path)
    fields = OrderedDict(fields)
    fields.update({
        'file': open(file_name, 'rb')
    })
    response = requests.post(url=url, files=fields, proxies=proxies)
    response.raise_for_status()


def download_file(url: str, timeout=600):
    response = requests.get(url=url, timeout=timeout, proxies=proxies)
    response.raise_for_status()
    return response.content


def check_md5(model_setting):
    model_path = model_setting.MODEL_PATH
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        if file_hash == model_setting.MODEL_HASH:
            return True
    return False


def get_model_tag(model_setting):
    model_tag = model_setting.MODEL_NAME
    if hasattr(model_setting, 'MODEL_VERSION'):
        version = model_setting.MODEL_VERSION
        if version != "v0":
            model_tag = "{}_{}".format(model_tag, version)
    return model_tag


def sync_model(model_setting, session=None, timeout=600):
    model_name = model_setting.MODEL_NAME
    model_path = model_setting.MODEL_PATH
    model_version = model_setting.MODEL_VERSION
    if check_md5(model_setting):
        logger.info('{}({}): md5 check success'.format(model_name, model_version))
    else:
        logger.info('{}({}): md5 check failed'.format(model_name, model_version))

        model_tag = get_model_tag(model_setting)
        if not session:
            with requests.session() as session:
                url = get_s3_url(session, project_name, model_tag, 'download')
        else:
            url = get_s3_url(session, project_name, model_tag, 'download')

        start_time = time.time()
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            tmpfilename = f.name
            try:
                content = download_file(url['ret']['url'], timeout=timeout)
                f.write(content)
            except Exception as e:
                logger.error(f'The downloaded file is error {str(e)}, cloud not exist this model {model_tag}')
                return

        model_archive_path = model_setting.MODEL_ARCHIVE_PATH
        if model_archive_path is not None:
            if not zipfile.is_zipfile(tmpfilename):
                logger.error('The downloaded file is not a valid zip file')
            else:
                model_dir = os.path.dirname(model_archive_path)
                with zipfile.ZipFile(tmpfilename, 'r') as zipf:
                    zipf.extractall(path=model_dir)
        else:
            # check and create destination directory
            model_dir = os.path.dirname(model_path)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            shutil.move(tmpfilename, model_path)

        end_time = time.time()
        logger.info('{}: downloaded file in {:.03f}'.format(
            model_name, end_time - start_time))

        if not check_md5(model_setting):
            logger.info('{}: download file failed'.format(model_name))


def batch_upload(model_settings=None):
    if not model_settings:
        model_settings = ModelSettings.get_models()

    logger.info('Upload the following models: {}'.format(
        ['{}({})'.format(m.MODEL_NAME, m.MODEL_VERSION) for m in model_settings]
    ))

    with requests.session() as session:
        for model_setting in model_settings:
            model_tag = get_model_tag(model_setting)
            upload_file_path = model_setting.MODEL_ARCHIVE_PATH
            if upload_file_path:
                if not os.path.exists(upload_file_path):
                    if os.path.splitext(upload_file_path)[-1] == '.zip':
                        if os.path.exists(os.path.splitext(upload_file_path)[0]):
                            zip_obj(upload_file_path, os.path.splitext(upload_file_path)[0])
                        else:
                            logger.error('file not found: {}'.format(upload_file_path))
                            continue
                    else:
                        upload_file_path = model_setting.MODEL_PATH
            else:
                upload_file_path = model_setting.MODEL_PATH

            if not os.path.exists(upload_file_path):
                logger.error('file not found: {}'.format(upload_file_path))
                continue

            # TODO: check if model exists

            url = get_s3_url(session, project_name, model_tag, 'upload')
            logger.info(f'uploading {upload_file_path}')
            start_time = time.time()
            upload_file(url['ret']['url']['url'], upload_file_path,
                        url['ret']['url']['fields'])
            end_time = time.time()
            logger.info('uploading {model_tag} done ({elapsed:.03f})'.format(
                model_tag=model_tag, elapsed=end_time - start_time))


def batch_sync():
    with requests.session() as session:
        for model_setting in ModelSettings.get_models():
            sync_model(model_setting, session)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synchronize model files')
    parser.add_argument('-u', '--upload', action='store_true',
                        help='Upload model files instead of synchronization.')
    parser.add_argument('-d', '--download', action='store_true',
                        help='download model files instead of synchronization.')
    args = parser.parse_args()

    if args.download:
        # 下载模型到本地
        batch_sync()

    if args.upload:
        # 上传模型到云端
        batch_upload()

    # 默认测试单个文件上传下载
    model_setting = default_model_setting._replace(
        MODEL_NAME='zawu_classifier',
        MODEL_VERSION='v11',
        MODEL_PATH='assets/zawu_cls_model_v11.pb',
    )
    with requests.session() as session:
        sync_model(model_setting, session)
    batch_upload([model_setting])
