import json

import os

from django.shortcuts import render
from django.shortcuts import redirect
from django.shortcuts import HttpResponse
from django.shortcuts import render_to_response

from django.contrib.auth.decorators import login_required

from django.template import RequestContext

from django.http import HttpResponseRedirect

from .models import Category
from .models import Project
from .models import Task
from .models import Member
from .models import Workspace
from UserProfile.models import UserMessage

from .forms import ProjectForm
from .forms import CategoryForm
from .forms import TaskForm
from UserProfile.forms import UpdateUserProfileForm
from UserProfile.forms import SendUserMessageForm

from django.core import serializers
from rest_framework import serializers as drf_serializers
from rest_framework import viewsets

def index(request):
    print("## ProjecetManager App index(request) call")
    # model form
    taskform = TaskForm()
    projectform = ProjectForm()
    categoryform = CategoryForm()

    # projects
    projects = Project.objects.all()
    psw_names = []
    pw_names = []

    for project in projects:
        pws = project.project_workspaces.all()
        for pw in pws:
            pw_names += [pw.workspace_name]
        psw_names += [pw_names]
        pw_names = []

    #print(pw_names)
    #print(psw_names)

    psm_names = []
    pm_names = []
    for project in projects:
        pms = project.project_members.all()
        for pm in pms:
            pm_names += [pm.member_name]
        psm_names += [pm_names]
        pm_names = []

    # D3js graph representation in Category panel

    # Build paths inside the project like this: os.path.join(BASE_DIR, ...)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(BASE_DIR, 'static/graph_temp/miserables.json')
    task_graph_json = open(path)
    #task_graph_json = json.load(task_graph_json) # deserialises it
    task_graph_json = json.dumps(json.load(task_graph_json)) # json formatted string

    # Javascript sorting for member ranking
    members = Member.objects.all()

    if request.method == 'POST':
        print("## request.method POST")
        print(request.POST.keys())
        # ctrl panel tasklist add
        if 'taskform_submit' in request.POST.keys():
            if request.POST['taskform_submit'] == 'submit':
                print("## taskform_submit")
                print("## taskform validation : ", taskform.is_valid())
                taskform = TaskForm(request.POST)
                if taskform.is_valid():
                    task = taskform.save(commit=False)
                    task.save()
        if 'projectform_submit' in request.POST.keys():
            if request.POST['projectform_submit'] == 'submit':
                print("## projectform_submit")
                print("## projectform validation : ", projectform.is_valid())
                projectform = ProjectForm(request.POST)
                if projectform.is_valid():
                    project = projectform.save(commit=False)
                    project.save()
        if 'categoryform_submit' in request.POST.keys():
            if request.POST['categoryform_submit'] == 'submit':
                print("## categoryform_submit")
                print("## categoryform validation : ", categoryform.is_valid())
                categoryform = CategoryForm(request.POST)
                if categoryform.is_valid():
                    category = categoryform.save(commit=False)
                    category.save()
        # ctrl panel tasklist remove
        if 'delete_task' in request.POST.keys():
            if request.POST['delete_task'] == '1':
                print("inside delete!")
                if request.is_ajax():
                    print('index call')
                    # print(request.POST['task_pks'])
                    print(request.POST.get('ajax_differer'))

                    # ctr1-task-table1
                    if request.POST['ajax_differer'] == '1':
                        delete_pks = request.POST.getlist('task_pks[]')
                        for ids in delete_pks:
                            print(ids, " deleted.")
                            task = Task.objects.get(pk=ids)
                            task.delete()
                            # data = serializers.serialize('json', request.POST["task_pks"])
                        #ctxes = index_ctx_call()
                        # return render(request, 'ProjectManagerDir/index.html', ctxes)
                        #taskform=TaskForm()
                        #render_to_response('ProjectManagerDir/index.html',{'taskform':taskform})
                        return HttpResponse("done!")

                    if request.POST['ajax_differer'] == '2':
                        return HttpResponse()

    tasks = Task.objects.all().order_by('-task_priority')

    print('ctz')

    # Top navbar message view
    is_authenticated = request.user.is_authenticated()
    username = request.user.username
    if is_authenticated == True:
        messages = UserMessage.objects.filter(touser__user__username__exact=request.user.username).all()
        num_messages = messages.count()

    categories = Category.objects.all()
    workspaces = Workspace.objects.all()
    ctx = {
        'is_authenticated': is_authenticated,
        'username': username,
        'projects': projects,
        'workspaces': workspaces,
        'projects_workspace_names': psw_names,
        'projects_member_names': psm_names,
        'task_graph_json': task_graph_json,
        'filepath': path,
        'tasks': tasks,
        'members': members,
        'taskform': taskform,
        'projectform': projectform,
        'categoryform': categoryform,
        'categories' : categories,
    }

    if is_authenticated == True:
        ctx['messages'] =  messages
        ctx['num_messages']= num_messages

    # TODO: multiple selection
    if request.method == 'POST':
        if 'category-select' in request.POST.keys():
            if request.POST['category-select'] == 'submit':
                categories_selected = []
                print(request.POST['category-open-select'])
                print(type(request.POST['category-open-select']))
                for pk in list(request.POST['category-open-select']):
                    categories_selected.append(Category.objects.get(pk=int(pk)))
                print(categories_selected)
                ctx['categories_selected'] = categories_selected

        if 'workspace-select' in request.POST.keys():
            if request.POST['workspace-select'] == 'submit':
                workspaces_selected = []
                print(request.POST['workspace-open-select'])
                print(type(request.POST['workspace-open-select']))
                for pk in list(request.POST['workspace-open-select']):
                    workspaces_selected.append(Workspace.objects.get(pk=int(pk)))
                print(workspaces_selected)
                ctx['workspaces_selected'] = workspaces_selected

        if 'project-select' in request.POST.keys():
            if request.POST['project-select'] == 'submit':
                projects_selected = []
                print(request.POST['project-open-select'])
                print(type(request.POST['project-open-select']))
                for pk in list(request.POST['project-open-select']):
                    projects_selected.append(Project.objects.get(pk=int(pk)))
                print(projects_selected)
                ctx['projects_selected'] = projects_selected

                return render(request, 'ProjectManagerDir/index-native.html', ctx)

    return render(request, 'ProjectManagerDir/index-native.html', ctx)


@login_required
def user_profile(request):
    print("UserProfile called")
    current_user = request.user
    profile = current_user.userprofile
    form = UpdateUserProfileForm(instance=current_user.userprofile)

    #TODO: modelform initialize with model : Not right!
    # usr_msg = UserMessage(fromuser=request.user.userprofile)
    #new_message_form = SendUserMessageForm(initial={'fromuser': request.user.userprofile})

    #modelform initialize with argument
    new_message_form = SendUserMessageForm(initial={'fromuser': request.user.userprofile})

    if request.method == 'POST':
        print("##user_profile post method call")
        files_ = request.FILES
        print(files_)
        form = UpdateUserProfileForm(request.POST, files_, instance=current_user.userprofile)
        print(form.errors)
        print(form.is_valid())
        if form.is_valid():
            form.save()

    messages = UserMessage.objects.filter(touser__user__username__exact=request.user.username).all()
    num_messages = messages.count()
    ctx = {
        'is_authenticated': True,
        "username": current_user.username,
        "user": profile,
        "user_profile_form": form,
        "messages": messages,
        'num_messages': num_messages,
        'new_message_form': new_message_form,
    }
    return render(request, 'ProjectManagerDir/user-profile.html', ctx)


@login_required
def user_send_message(request):
    new_message_form = SendUserMessageForm()
    if request.method == 'POST':
        print("user_send_message wtih post!")
        new_message_form = SendUserMessageForm(request.POST)
        if new_message_form.is_valid():
            new_message_form.save()
            return HttpResponseRedirect(request.META.get('HTTP_REFERER'))
    return HttpResponseRedirect(request.META.get('HTTP_REFERER'))


@login_required
def delete_task(request):
        if request.method == 'POST':
            if request.yes == 1 :
                return