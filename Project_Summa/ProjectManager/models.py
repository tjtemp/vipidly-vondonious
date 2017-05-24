from __future__ import unicode_literals

from django.db import models
from django.db.models.signals import post_save
from django.db.models.signals import post_init
from django.dispatch import receiver

from decimal import Decimal

# Progress is determined when unitest problem is passed.
# each problem is assigned a value and it is summed up to progress.
class Progress(models.Model):
    progress_name = models.CharField(max_length=100, default="progress_")
    progress_project = models.ForeignKey('Project')
    # progress_task can be assigned to Project itself, or Task under project
    progress_task = models.ForeignKey('Task', null=True, blank=True)
    progress_member = models.ForeignKey('Member', null=True, blank=True)
    # only takes 99.99?
    progress_gauge = models.DecimalField(default=0.00, decimal_places=2, max_digits=3)

    progress_open_date = models.DateTimeField(auto_now=True)
    progress_closed_date = models.DateTimeField(blank=True, null=True)

    def __str__(self):
        return '{}-{}%'.format(self.progress_name, str(self.progress_gauge))

    class Meta:
        pass


class Task(models.Model):
    task_status = models.BooleanField(default=False)
    task_priority = models.IntegerField(default=1)
    # if null=True is added,
    # WARNINGS:
    # ProjectManager.Task.task_related: (fields.W340) null has no effect on ManyToManyField.
    task_related = models.ManyToManyField('self', blank=True)
    task_name = models.CharField(max_length=100, default="task_")
    task_description = models.CharField(max_length=300, null=True, blank=True)
    task_project = models.ForeignKey('Project')
    task_submitter = models.CharField(max_length=100, default="Anonymous", null=True, blank=True)
    # if null=True is added,
    # WARNINGS:
    # ProjectManager.Task.task_members: (fields.W340) null has no effect on ManyToManyField.
    task_members = models.ManyToManyField('Member', blank=True)
    task_initiated_at = models.DateTimeField(auto_now=True)
    task_deadline = models.DateTimeField(null=True, blank=True)
    task_completed_at = models.DateTimeField(null=True, blank=True)

    # cannot put progress_sum function to default with self argument.
    task_progress = models.DecimalField(default=0.00, decimal_places=2, max_digits=3, editable=True)

    def progress_sum(self):
        progresses = self.progress_set.all()
        print(progresses)
        progresses_gauge = 0
        for progress in progresses:
            progresses_gauge += progress.progress_gauge
        print(progresses_gauge)
        return progresses_gauge

    """
    this calls error!
    def save(self, *args, **kwargs):
        if not self.task_progress:
            self.task_progress = self.progress_sum()
            print(self.task_progress)
        super().save(*args, **kwargs)
    """

    def __str__(self):
        return '{}-{} : {}'.format(self.task_name, self.task_project, self.task_members)


import json
def UpdateGraph():
    print("UpdateGraph() Called.")
    dics = {'name': "project_summa", 'children': []}
    fp = open('static/js/ProjectManagerTree.json', 'w', encoding='utf8')
    categories = Category.objects.all()
    for category in categories:
        catdics = {}
        catdics['name'] = category.category_name
        catdics['children'] = []

        projects = category.project_set.all()
        for project in projects:
            prodics = {}
            prodics['name'] = project.project_name
            prodics['children'] = []

            tasks = project.task_set.all()
            for task in tasks:
                tasdics = {}
                tasdics['name'] = task.task_name
                tasdics['children'] = []
                prodics['children'].append(tasdics)
            catdics['children'].append(prodics)
        dics['children'].append(catdics)
    json.dump(dics, fp, indent=4)
    fp.close()


@receiver(post_save, sender=Task)
def post_save_task(sender, instance, created, **kwargs):
    if created:
        #instance = kwargs.pop('instance')
        instance.task_progress = instance.progress_sum()
        instance.save(update_fields=["task_progress"])
        UpdateGraph()





class Member(models.Model):
    """
        TODO: currently this is not instantiated as UserProfile created. untied Member represents T.O.
    """
    member_name = models.CharField(max_length=100, default="member_", help_text="project member nickname")
    member_department = models.CharField(max_length=100 ,
                                         choices=((u'project manager',u'project manager'),
                                                  (u'team manager',u'team manager'),
                                                  (u'freelancer',u'freelancer'),
                                                  (u'backend programmer', u'backend programmer'),
                                                  (u'frontend programmer', u'frontend programmer'),
                                                  (u'undesignated', u'undesignated')),
                                         default="undesignated")

    def __str__(self):
        return '{}-{}'.format(self.member_name, self.member_department)


class Project(models.Model):
    project_status = models.BooleanField(default=False, help_text='True: Done, False: Not Done')
    project_name = models.CharField(max_length=100, default="project_")
    project_members = models.ManyToManyField('Member')
    project_workspaces = models.ManyToManyField('Workspace')
    project_controller = models.ForeignKey('Controller')
    project_category = models.ForeignKey('Category')

    project_start_date = models.DateTimeField(auto_now=True)
    project_end_date = models.DateTimeField(null=True, blank=True)
    project_deadline = models.DateTimeField(null=True, blank=True)

    project_description = models.CharField(max_length=300, null=True,
                                           default="test project_", blank=True)

    def __str__(self):
        return '{} - {} name: {} at {}'.format(
            self.project_controller,
            [workspace.workspace_name for workspace in self.project_workspaces.all()],
            self.project_name,
            self.project_start_date)

    def get_absolute_url(self):
        pass

    def describe(self):
        return '{}'.format(self.workspace_description)

    class Meta:
        pass


class Category(models.Model):
    category_parent = models.ForeignKey('self', null=True, blank=True)
    category_name = models.CharField(max_length=100, default="category_")

    def __str__(self):
        return 'category - {}'.format(self.category_name)


class Workspace(models.Model):
    workspace_name = models.CharField(max_length=100, default="workspace_")
    workspace_controller = models.ForeignKey('Controller')
    workspace_description = models.CharField(max_length=300, default="test workspace_",
                                             null="True", blank=True)
    def __str__(self):
        return 'workspace - {}'.format(self.workspace_name)

    def describe(self):
        return '{}'.format(self.workspace_description)


class Controller(models.Model):
    controller_name = models.CharField(max_length=100, default="controller_")
    controller_description = models.CharField(max_length=300, default="test workspace_")

    def __str__(self):
        return '{}'.format(self.controller_name)

    def describe(self):
        return '{}'.format(self.controller_description)