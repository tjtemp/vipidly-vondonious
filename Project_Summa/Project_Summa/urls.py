"""Project_Summa URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.10/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.conf.urls import include

from django.contrib import admin

from django.contrib.auth.views import login
from django.contrib.auth.views import logout
from django.contrib.auth import views as auth_views

from django.conf import settings
from django.conf.urls.static import static

import django.views.defaults

# Do not put $ at the end of url if there's more.
urlpatterns = [
    url(r'^', include('Home.urls'), name="home"),
    url(r'^admin/', admin.site.urls),
    url(r'^analyzer/', include('Analyzer.urls'), name="analyzer"),
    url(r'^projectmanager/', include('ProjectManager.urls'), name="projectmanager"),
    url(r'^contact/', include('Contact.urls'), name="contact"),
    url(r'^testapp/', include('testapp.urls'), name="testapp")
#    url(r'^404/$', django.views.defaults.page_not_found), not working
]

handler404 = 'Home.views.handler404'
handler403 = 'Home.views.handler403'

urlpatterns += static(
    settings.MEDIA_URL, document_root=settings.MEDIA_ROOT,
)
