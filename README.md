## tools:

### test.py
test sync models to local, test local model upload to cloud
+ `python test.py`
test download single model and upload single model
+ `python test.py -u`
add upload all models to cloud
+ `python test.py -d`
add download all models from cloud
+ `python test.py -u -d`
add upload and download models

### model_config.py
generte model name, version, save path mapping relationship to ***name_path_map.json***, need move ***name_path_map.json*** to base_dir

### additional_model_settings.json
user add new model not in model_config.py

### model_config.json
user modify model settings in model_config.py

### local_example.py
nextcloud api function test case code


## dev.evns
required environment variables


## web api:
|uri|method|request|response|
|---|---|---|---|
|https://cv.turingvideo.cn/overlord/user/login|POST|
{
    'email': 'guardian+m3@turingvideo.com',
    'password': ')S1I!2<:Uq=j'
}|200|
|https://cv.turingvideo.cn/scv/s3url/model/{type}/{project}/{name}|GET||
'download':
    {
        'ret': {
            'url': url
        }
    }
'upload':
    {
        'ret': {
            'url': {
                'url': url,
                'fields': {}
            }
        }
    }
|
|https://cv.turingvideo.cn/download/<str:project>/<str:name>|GET||open('xxx', 'rb')|
|https://cv.turingvideo.cn/upload/<str:project>/<str:name>|POST|
fields = {
    'file': open(file_name, 'rb')
}
|200|
