from django.contrib import admin

from .models import Progress
from .models import Task
from .models import Member
from .models import Project
from .models import Category
from .models import Workspace
from .models import Controller


class ProgressAdmin(admin.ModelAdmin):

    list_display = ('progress_name', 'progress_project', 'progress_task', )
    search_fields = ('progress_name', 'progress_project', 'progress_task', )


#class ProjectAdmin(admin.ModelAdmin):

#     list_display = ('project_name')
#     search_fields = ('project_name', 'project_members__id', 'project_workspace__id' )

# In [28]: Project.objects.all().prefetch_related('project_workspaces')
# Out[28]: <QuerySet [<Project: controller_main - ['porfolio'] name: summa at 2016-12-13 05:30:19.944600+00:00>, <Project: controller_main - ['porfolio'] name: pystagram at 2016-12-13 05:30:11.332688+00:0
# 0>, <Project: controller_main - ['porfolio'] name: myplace at 2016-12-13 06:42:11.022357+00:00>]>

# In [29]: Project.objects.get(pk=1).project_workspaces.all()
# Out[29]: <QuerySet [<Workspace: workspace - porfolio>]>

admin.site.register(Progress, ProgressAdmin)
admin.site.register(Member)
admin.site.register(Task)
admin.site.register(Project)#, ProjectAdmin)
admin.site.register(Category)
admin.site.register(Workspace)
admin.site.register(Controller)