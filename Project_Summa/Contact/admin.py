from django.contrib import admin

from .models import Category
from .models import Post
from .models import Comment
from .models import Tag


class PostAdmin(admin.ModelAdmin):
    list_display = ('id', 'category', 'title', 'user', 'created_at')
    search_fields = ('category', 'title', 'user', )

admin.site.register(Category)
admin.site.register(Post, PostAdmin)
admin.site.register(Comment)
admin.site.register(Tag)

