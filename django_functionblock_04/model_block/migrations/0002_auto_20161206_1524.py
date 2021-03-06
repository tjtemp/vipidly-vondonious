# -*- coding: utf-8 -*-
# Generated by Django 1.10.3 on 2016-12-06 06:24
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('model_block', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='firstmodel',
            name='modelfield7',
        ),
        migrations.AddField(
            model_name='firstmodel',
            name='modelfeild7_2',
            field=models.NullBooleanField(),
        ),
        migrations.AddField(
            model_name='firstmodel',
            name='modelfield7_1',
            field=models.BooleanField(default=True),
        ),
        migrations.AlterField(
            model_name='firstmodel',
            name='modelfield1_1',
            field=models.IntegerField(null=True),
        ),
        migrations.AlterField(
            model_name='firstmodel',
            name='modelfield1_2',
            field=models.BigIntegerField(null=True, unique=True, verbose_name='64bit integer'),
        ),
        migrations.AlterField(
            model_name='firstmodel',
            name='modelfield1_3',
            field=models.PositiveIntegerField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='firstmodel',
            name='modelfield1_4',
            field=models.SmallIntegerField(null=True, verbose_name='16bit integer'),
        ),
        migrations.AlterField(
            model_name='firstmodel',
            name='modelfield1_5',
            field=models.FloatField(choices=[(1.1, 'first'), (2.1, 'second'), (3.1, 'third')], null=True),
        ),
        migrations.AlterField(
            model_name='firstmodel',
            name='modelfield1_6',
            field=models.DecimalField(decimal_places=4, max_digits=10, null=True),
        ),
        migrations.AlterField(
            model_name='firstmodel',
            name='modelfield2',
            field=models.BinaryField(null=True),
        ),
        migrations.AlterField(
            model_name='firstmodel',
            name='modelfield3_2',
            field=models.EmailField(max_length=254, null=True),
        ),
        migrations.AlterField(
            model_name='firstmodel',
            name='modelfield3_3',
            field=models.URLField(null=True),
        ),
        migrations.AlterField(
            model_name='firstmodel',
            name='modelfield3_4',
            field=models.TextField(null=True),
        ),
        migrations.AlterField(
            model_name='firstmodel',
            name='modelfield4_1',
            field=models.FileField(null=True, upload_to=''),
        ),
        migrations.AlterField(
            model_name='firstmodel',
            name='modelfield4_2',
            field=models.ImageField(auto_created=True, null=True, upload_to='%yy/%mm/%dd/'),
        ),
        migrations.AlterField(
            model_name='firstmodel',
            name='modelfield6',
            field=models.GenericIPAddressField(null=True),
        ),
    ]
