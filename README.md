tools:

test.py:  test sync models to local, test local model upload to cloud
1、 python test.py  ----test download single model and upload single model
2、 python test.py -u  ----add upload all models to cloud
3、 python test.py -d  ----add download all models from cloud
4、 python test.py -u -d ----add upload and download models

model_config.py: generte model name, version, save path mapping relationship to name_path_map.json, need move name_path_map.json to base_dir
additional_model_settings.json: user add new model not in model_config.py
model_config.json: user modify model settings in model_config.py
local_example.py: nextcloud api function test case code


dev.evns:
required environment variables


web api:
overlord/user/login: just response login
scv/s3url/model/<str:type>/<str:project>/<str:name>: return download or upload url
download/<str:project>/<str:name>: download model from nextcloud
upload/<str:project>/<str:name>': upload model to nextcloud
