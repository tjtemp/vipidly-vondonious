# -*- coding: utf-8 -*-
# Generated by Django 1.10.5 on 2017-02-15 07:45
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ProjectManager', '0009_progress_progress_open_date'),
    ]

    operations = [
        migrations.AddField(
            model_name='progress',
            name='progress_closed_date',
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]
