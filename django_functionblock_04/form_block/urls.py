from django.conf.urls import url


from .views import index
from . import views

from .views import modal_form_index
from .views import modal_form_result


app_name = 'form_block'

urlpatterns = [
    #/form_block/
    url(r'^$', index, name='index'),
    url(r'^modal_form_index', modal_form_index, name='modal_form_index'),
    url(r'^basic-upload/$', views.BasicUploadView.as_view(), name='basic_upload'),
    url(r'^modal_form_result', modal_form_result, name='modal_form_result'),
]