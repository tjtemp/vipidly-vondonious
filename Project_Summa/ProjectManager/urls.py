from django.conf.urls import url

from .views import index
from .views import delete_task
from .views import user_profile
from .views import user_send_message

app_name = "projectmanager"

urlpatterns = [
    # /projectmanager/
    url(r'^$', index , name = "index"),
    url(r'^user-profile', user_profile, name = "profile"),
    url(r'^send-message', user_send_message, name = "sendmessage"),
    url(r'^delete-task', delete_task, name = "deletetask")
]