# -*- coding: utf-8 -*-
# Generated by Django 1.10.4 on 2017-02-06 16:23
from __future__ import unicode_literals

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('UserProfile', '0005_remove_usermessage_received'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='usermessagereceived',
            name='fromuser',
        ),
        migrations.RemoveField(
            model_name='usermessagereceived',
            name='received',
        ),
        migrations.RemoveField(
            model_name='usermessagereceived',
            name='user',
        ),
        migrations.RemoveField(
            model_name='usermessagesent',
            name='touser',
        ),
        migrations.RemoveField(
            model_name='usermessagesent',
            name='user',
        ),
        migrations.DeleteModel(
            name='UserMessageReceived',
        ),
        migrations.DeleteModel(
            name='UserMessageSent',
        ),
    ]
