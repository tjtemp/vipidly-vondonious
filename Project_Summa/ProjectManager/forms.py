from django import forms

from .models import Category
from .models import Project
from .models import Task


class CategoryForm(forms.ModelForm):
    class Meta:
        model = Category
        fields = '__all__'

class ProjectForm(forms.ModelForm):
    class Meta:
        model = Project
        fields = '__all__'


class TaskForm(forms.ModelForm):
    class Meta:
        model = Task
        fields = '__all__'

