from django.conf.urls import url
from django.contrib import admin

from .views import list_post
from .views import view_post
from .views import create_post
from .views import delete_comment

from .views import index

app_name = 'contact'

urlpatterns = [
    # /contact/
    url(r'^$', index, name="index"),
    url(r'^list/$', list_post, name="list"),
    url(r'^list/(?P<pk>[0-9]+)/$', view_post, name="view"),
    url(r'^list/new/$', create_post, name="new"),
    url(r'comments/(?P<pk>[0-9]+)/delete$', delete_comment, name="delete_comment"),
]
