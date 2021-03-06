# -*- coding: utf-8 -*-
# Generated by Django 1.10.5 on 2017-04-16 15:37
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Analyzer', '0007_job_job_submitter'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='datafile',
            name='filepath',
        ),
        migrations.RemoveField(
            model_name='image',
            name='image_file',
        ),
        migrations.RemoveField(
            model_name='image',
            name='image_name',
        ),
        migrations.RemoveField(
            model_name='image',
            name='image_owner',
        ),
        migrations.RemoveField(
            model_name='video',
            name='video_file',
        ),
        migrations.RemoveField(
            model_name='video',
            name='video_name',
        ),
        migrations.RemoveField(
            model_name='video',
            name='video_owner',
        ),
        migrations.AddField(
            model_name='image',
            name='image_type',
            field=models.CharField(default='image_', max_length=20),
        ),
        migrations.AlterField(
            model_name='video',
            name='video_type',
            field=models.CharField(default='video_', max_length=20),
        ),
    ]
