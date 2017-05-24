from django.contrib import admin

from .models import UserProfile
from .models import UserMessage

class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'projectinfo')
    search_fields = ('user', 'projectinfo')


class UserMessageAdmin(admin.ModelAdmin):
    list_display = ('title', 'fromuser', 'touser', 'created_at')
    search_fields = ('title', 'created_at')


admin.site.register(UserProfile, UserProfileAdmin)
admin.site.register(UserMessage, UserMessageAdmin)