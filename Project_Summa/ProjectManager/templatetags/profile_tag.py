from django import template
from ProjectManager.models import Progress

from decimal import Decimal

register = template.Library()

@register.filter(name='check_profile_photo_bool')
def check_profile_photo_bool(profile_photo_field):
    return bool(profile_photo_field)


@register.filter(name='member_progress_sum')
def member_progress_sum(member):
    val=0
    for gauges in Progress.objects.filter(progress_member__exact=member):
        val += gauges.progress_gauge
    return val


@register.filter(name='project_select_by_category_name')
def project_select_by_category_name(project):
    return None


@register.filter(name='project_progress_sum')
def project_progress_sum(project):
    val=0
    for gauges in Progress.objects.filter(progress_project__exact=project):
        val += gauges.progress_gauge
    return val

@register.filter(name='project_in_workspace')
def projects_in_workspace(workspace):
    return workspace.project_set.all()

@register.filter(name='get_tasks_in_project')
def get_tasks_in_project(project):
    return project.task_set.all()

@register.filter(name='get_members_in_task')
def get_members_in_task(task):
    return task.task_members.all()

@register.filter(name='get_member_gauge_in_task')
def get_member_gauge_in_task(task, member):
    # get function query does not exist error if query is empty
    #task.progress_set.get(progress_member__exact=member).progress_gauge
    if task.progress_set.filter(progress_member__exact=member).exists():
        return task.progress_set.get(progress_member__exact=member).progress_gauge
    else:
        return 0

@register.filter(name='get_member_opendate_in_task')
def get_member_opendate_in_task(task, member):
    if task.progress_set.filter(progress_member__exact=member).exists():
        return task.progress_set.get(progress_member__exact=member).progress_open_date
    else:
        return None

# TODO: check contains function is also valid in list form
@register.filter(name='member_project_check')
def member_project_check(member, project):
    name = member.member_name
    return project.project_members.filter(member_name__contains=name).exists()


# TODO: single workspace only
@register.filter(name='length_range')
def length_range(workspace):
    print("length_range!")
    print(type(workspace))
    print(workspace)
    if hasattr(workspace, '__iter__'):
        print("list!")
        strings=""
        for idx in range(len(workspace[0].project_set.all())):
            strings += str(idx)
        return strings
    else:
        return "0"

# TODO: single project only : currently only return 0
@register.filter(name='length_project_range')
def length_project_range(projects):

    if hasattr(projects, '__iter__'):
        print("list!")
        strings=""
        for idx in range(len(projects)):
            strings += str(idx)
        return strings
    else:
        return "0"

@register.filter(name='mul_100')
def mul_100(nums):
    return nums*100

@register.assignment_tag
def alias(obj):
    """
    Alias Tag
    """
    return obj


class SetVarNode(template.Node):

    def __init__(self, var_name, var_value):
        self.var_name = var_name
        self.var_value = var_value

    def render(self, context):
        try:
            value = template.Variable(self.var_value).resolve(context)
        except template.VariableDoesNotExist:
            value = ""
        context[self.var_name] = value

        return u""


@register.tag(name='set')
def set_var(parser, token):
    """
    {% set some_var = '123' %}
    """
    parts = token.split_contents()
    if len(parts) < 4:
        raise template.TemplateSyntaxError("'set' tag must be of the form: {% set <var_name> = <var_value> %}")

    return SetVarNode(parts[1], parts[3])