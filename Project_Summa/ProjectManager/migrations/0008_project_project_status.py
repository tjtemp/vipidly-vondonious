# -*- coding: utf-8 -*-
# Generated by Django 1.10.4 on 2017-02-13 14:22
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ProjectManager', '0007_auto_20170213_1338'),
    ]

    operations = [
        migrations.AddField(
            model_name='project',
            name='project_status',
            field=models.BooleanField(default=False, help_text='True: Done, False: Not Done'),
        ),
    ]