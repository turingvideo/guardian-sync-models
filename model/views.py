# import json
import os
import tempfile
import logging
from django.http import HttpResponse, JsonResponse, FileResponse, Http404

from .service import download_file, upload_file


_logger = logging.getLogger(__name__)


def login(request):
    if request.method == 'POST':
        return HttpResponse("login success!")
    else:
        raise Http404


def s3_url(request, type, project, name):
    absolute_uri = request.build_absolute_uri(request.get_full_path())
    url = absolute_uri.replace(request.get_full_path(), f'/{type}/{project}/{name}')
    if type == 'download':
        res = {
            'ret': {
                'url': url
            }
        }
    elif type == 'upload':
        res = {
            'ret': {
                'url': {
                    'url': url,
                    'fields': {}
                }
            }
        }
    else:
        res = {}
    return JsonResponse(res)


def download(request, project, name):
    try:
        _logger.info(f'download project:{project}, name:{name}')
        download_file_path = download_file(project, name)
        response = FileResponse(open(download_file_path, 'rb'))
        response['content-type'] = 'application/octet-stream'
        response['Content-Disposition'] = 'attachment; filename=' + os.path.basename(download_file_path)
        if os.path.exists(download_file_path):
            os.remove(download_file_path)
        return response
    except Exception:
        raise Http404


def upload(request, project, name):
    _logger.info(f'upload project:{project}, name:{name}')
    my_file = request.FILES['file']
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        tmpfilename = f.name
        for chunk in my_file.chunks():
            f.write(chunk)
    upload_file(tmpfilename, project, name)
    if os.path.exists(tmpfilename):
        os.remove(tmpfilename)
    return JsonResponse({'method': 'post', 'function': 'upload'})
