# -*- coding: utf-8 -*-
# Generated by Django 1.10.4 on 2017-01-26 16:40
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Analyzer_core',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('analyzer_name', models.CharField(max_length=50)),
            ],
            options={
                'ordering': ('-pk',),
            },
        ),
        migrations.CreateModel(
            name='Controller',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('controller_name', models.CharField(blank=True, max_length=40)),
                ('controller_problem_settings', models.CharField(choices=[('classification', 'classification'), ('regression', 'regression'), ('clustering', 'clustering')], default='classification', max_length=40)),
                ('controller_core_packages', models.CharField(blank=True, choices=[('sklearn', 'sklearn'), ('lasagne', 'lasagne'), ('keras', 'keras'), ('theano', 'theano'), ('tensorflow', 'tensorflow'), ('torch', 'torch'), ('caffe', 'caffe')], default='scikit-learn', max_length=50)),
                ('controller_core_estimator', models.CharField(blank=True, choices=[('LogisticRegression', 'LogisticRegression'), ('SVM', 'SVM'), ('XgBoost', 'XgBoost'), ('AlexNet', 'AlexNet'), ('GoogleNet', 'GoogleNet'), ('ResNet', 'ResNet')], default='LogisticRegression', max_length=50)),
                ('controller_core_parameters', models.CharField(blank=True, max_length=100)),
                ('controller_visual_tool', models.CharField(choices=[('scatter', 'scatter')], default='scatter', max_length=100)),
                ('controller_input_file_field', models.FileField(blank=True, null=True, upload_to='')),
                ('controller_created_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'ordering': ('-pk',),
            },
        ),
        migrations.CreateModel(
            name='Image',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image_name', models.CharField(max_length=100)),
                ('image_file', models.ImageField(upload_to='images/')),
            ],
        ),
        migrations.CreateModel(
            name='PostProcessor',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('postprocessor_name', models.CharField(default='test_postprocess', max_length=40)),
            ],
        ),
        migrations.CreateModel(
            name='PreProcessor',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('preprocessor_name', models.CharField(default='test_preprocess', max_length=40)),
            ],
        ),
        migrations.CreateModel(
            name='Video',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('video_name', models.CharField(max_length=100)),
                ('video_type', models.CharField(max_length=50)),
                ('video_file', models.FileField(upload_to='videos/')),
            ],
        ),
    ]