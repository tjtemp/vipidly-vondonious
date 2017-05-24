from django.shortcuts import render

from django.contrib.auth.decorators import login_required

from ProjectManager.models import Task, Project
#from Analyzer.models import UserImage, UserVideo

# UserProfile doesn't have view.

@login_required
def get_user_profile_folder(instance, filename):
    return "%s/info/%s" % (instance.user.username, filename)


@login_required
def get_user_file_folder(instance, filename):
    return "%s/file/%s" % (instance.user.username, filename)


@login_required
def get_user_imagefile_folder(instance, filename):
    return "%s/imagefile/%s" % (instance.user.username, filename)


@login_required
def get_user_profile_folder_filename(instance, _):
    """
    TODO:
    Currently this path is hardcoded in index, base in ProjectManager and Analyzer Apps.
    Changes should be in accordance with them.
    :param instance:
    :param _:
    :return:
    """
    return "%s/info/profile-photo_%s_%s.jpg" % (instance.user.username, instance.user.username, instance.user.pk)


@login_required
def get_user_upload_profile_photo_callback(instance, filename):
    def get_user_profile_folder_filename(instance, _):
        return "%s/info/profile-photo_%s_%s" % (instance.user.username, instance.user.username, instance.user.pk)
    return get_user_profile_folder_filename


@login_required
def get_user_uploadfile_folder(fileclass, filename):
    '''
    :param upload_apps: classname such as Projects or Task..
    :param filename:
    :return:
    NOTE: FileField has builtin function ._get_path()
    '''
    if isinstance(fileclass, Project):
        return "ProjectManager/%s/%s" % (fileclass.project_name, filename)
    if isinstance(fileclass, Task):
        return "ProjectManager/%s/%s/%s" % (fileclass.task_project.project_name, fileclass.task_name, filename)

    # elif isinstance(fileclass, UserImage): # User file on user folders
    #     return "%s/%s/%s" % (fileclass.image_owner.username, 'image', filename)
    # elif isinstance(fileclass, UserVideo):
    #     return "%s/%s/%s" % (fileclass.image_owner.username, 'video', filename)

