import os
from nextcloud import NextCloud

NEXTCLOUD_URL = 'http://localhost'
NEXTCLOUD_USERNAME = 'admin'
NEXTCLOUD_PASSWORD = '123456'
to_js = True

nxc = NextCloud(endpoint=NEXTCLOUD_URL, user=NEXTCLOUD_USERNAME, password=NEXTCLOUD_PASSWORD, json_output=to_js)
# 获取用户的列表
a = nxc.get_users()
print(a.data)


# 获取用户的文件夹信息
c = nxc.list_folders('admin')
print(c.data)
print(c.is_ok)

# 上传图片
local_filepath = '/Users/liuxushun/lxs.png'
upload_filepath = 'Photos/lxs.png'

print('11111111111111')
b = nxc.upload_file('admin', local_filepath, upload_filepath)
print(b.data, b.raw, b.is_ok)

# 分享图片拿到公共链接
d = nxc.create_share('Photos/lxs.png', 3)
print(d.data, d.raw)

# 下载图片
nxc.download_file('admin', 'Photos/Frog.jpg')
print('++++++++++++')
os.remove('Frog.jpg')

# 删除图片
f = nxc.delete_path('admin', upload_filepath)
print(f, type(f), type(f.data), f.raw, f.is_ok)

# 列出文件夹信息
g = nxc.list_folders('admin', path=None)
print(g.data)
print('---------------')
h = nxc.list_folders('admin', path='Photos')
print(h.data)

# 判断是否是文件夹
for ff in g.data:
    print(type(ff))
    if ff['resource_type'] == 'collection':
        print(ff['href'])


print(os.path.dirname(os.path.abspath(__file__)))
d = nxc.download_file('admin', 'assets/zawu_cls_model_v11.pb')
print(d)
