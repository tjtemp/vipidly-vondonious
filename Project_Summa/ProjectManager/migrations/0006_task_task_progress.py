# -*- coding: utf-8 -*-
# Generated by Django 1.10.4 on 2017-02-12 05:11
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ProjectManager', '0005_auto_20170212_0343'),
    ]

    operations = [
        migrations.AddField(
            model_name='task',
            name='task_progress',
            field=models.DecimalField(decimal_places=1, default=0, editable=False, max_digits=3),
        ),
    ]
