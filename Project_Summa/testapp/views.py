from django.shortcuts import render

import base64

from Analyzer.models import Datafile
from Analyzer.forms import datafile_upload_model_form


def index(request):
    ctx={}
    if request.method == 'POST':
        print('## request method POST')
        snapshot = Datafile()
        if request.is_ajax():
            img = request.POST.get('imgBase64')
            #img = get_base64_image(img)
            snapshot.filepath = img
            snapshot.save()
    return render(request, 'index.html', ctx)


def get_base64_image(data):
    if data is None or ';base64' not in data:
        return None
    _format, _content = data.split(';base64,')
    return base64.b64decode(_content)