from __future__ import unicode_literals

from django.db import models

from django.urls import reverse_lazy

from django.conf import settings


class Category(models.Model):
    name = models.CharField(max_length=40)
    parent = models.ForeignKey('self', null=True, blank=True)

    def __str__(self):
        return '{}'.format(self.name)


class Post(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL)
    title = models.CharField(max_length=100, blank=False, default="no title")
    category = models.ForeignKey(Category)
    image = models.ImageField(upload_to='%Y/%m/%d/', null=True, blank=True)
    content = models.TextField(null=False, blank=False)
    tags = models.ManyToManyField('Tag', blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return '{}'.format(self.pk)

    def get_absolute_url(self):
        return reverse_lazy('contact:view', kwargs={'pk': self.pk})


class Comment(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL)
    post = models.ForeignKey(Post)
    content = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']


class Tag(models.Model):
    name = models.CharField(max_length=40)

    def __str__(self):
        return '{}'.format(self.name)
