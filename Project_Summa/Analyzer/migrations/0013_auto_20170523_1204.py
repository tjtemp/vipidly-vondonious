# -*- coding: utf-8 -*-
# Generated by Django 1.11.1 on 2017-05-23 12:04
from __future__ import unicode_literals

import Analyzer.models
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Analyzer', '0012_auto_20170520_0953'),
    ]

    operations = [
        migrations.AlterField(
            model_name='datafile',
            name='filepath',
            field=models.FileField(blank=True, max_length=500, null=True, upload_to=Analyzer.models.get_user_uploadfile_folder),
        ),
    ]
