# -*- coding: utf-8 -*-
# Generated by Django 1.10.5 on 2017-02-15 07:43
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ProjectManager', '0008_project_project_status'),
    ]

    operations = [
        migrations.AddField(
            model_name='progress',
            name='progress_open_date',
            field=models.DateTimeField(auto_now=True),
        ),
    ]