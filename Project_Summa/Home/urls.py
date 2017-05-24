from django.conf.urls import url

from django.conf import settings

from django.conf.urls import include

from django.contrib.auth.views import login
from django.contrib.auth.views import logout

from django.contrib.auth import views as auth_views

from .views import index
from .views import UserLogin
from .views import UserSignupView
from .views import UserLogout

app_name='home'

urlpatterns = [
    # /home/
    url(r'^$', index, name = "index"),
    # default template folder is registration
    url(r'^login/$', UserLogin, name="login"),
    url(r'^signup/$', UserSignupView.as_view(), name="signup"),
    url(r'^logout/$', UserLogout, name='logout'),
    url(r'^analyzer/', include('Analyzer.urls'), name="analyzer"),
    url(r'^projectmanager/', include('ProjectManager.urls'), name="projectmanager"),
    url(r'^contact/', include('Contact.urls'), name="contact"),
    url(r'^testapp/', include('testapp.urls'), name="testapp"),
]