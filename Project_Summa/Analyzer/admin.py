from django.contrib import admin

from .models import Job

class JobAdmin(admin.ModelAdmin):
    list_display = ('job_status', 'job_priority', 'job_started_at', 'job_deadline', 'job_completed_at')
    search_fields = ('job_status', 'job_priority', 'job_started_at', 'job_deadline', 'job_completed_at')

admin.site.register(Job, JobAdmin)
