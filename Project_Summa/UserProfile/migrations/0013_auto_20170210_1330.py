# -*- coding: utf-8 -*-
# Generated by Django 1.10.4 on 2017-02-10 13:30
from __future__ import unicode_literals

import UserProfile.views
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('UserProfile', '0012_auto_20170210_1318'),
    ]

    operations = [
        migrations.AlterField(
            model_name='userprofile',
            name='profile_photo',
            field=models.FileField(blank=True, null=True, upload_to=UserProfile.views.get_user_profile_folder_filename),
        ),
    ]
