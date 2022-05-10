## tools

### test.py
test sync models to local, test local model upload to cloud  
`python test.py`   
test download single model and upload single model  
`python test.py -u`   
add upload all models to cloud  
`python test.py -d`  
add download all models from cloud  
`python test.py -u -d`  
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


## web api
#### https://cv.turingvideo.cn/overlord/user/login
***request***  
method: POST  
json:  
``` 
{
    'email': 'guardian+m3@turingvideo.com',
    'password': ')S1I!2<:Uq=j'
}
```
***response***  
200  

#### https://cv.turingvideo.cn/scv/s3url/model/{type}/{project}/{name}
***request***  
method: GET  
***response***
if type == download, return  
```
{
    'ret': {
        'url': url
    }
}
```
if type == upload, return  
```
{
    'ret': {
        'url': {
            'url': url,
            'fields': {}
        }
    }
}
```

#### https://cv.turingvideo.cn/download/\<str:project\>/\<str:name\>
***request***  
method: GET  
***response***  
open('xxx', 'rb')  

#### https://cv.turingvideo.cn/upload/\< str:project\ >/\< str:name \>
***request***  
method: POST  
fields:  
```
{
    'file': open(file_name, 'rb')
}
```
***response***
200
