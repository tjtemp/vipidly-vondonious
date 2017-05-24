from django.conf.urls import url

from .views import index

app_name = "testapp"

urlpatterns = [
    url(r'^', index)
]