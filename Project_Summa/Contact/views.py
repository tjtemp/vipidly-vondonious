import os

from django.shortcuts import render
from django.shortcuts import redirect
from django.shortcuts import get_object_or_404

from django.core.paginator import Paginator
from django.core.paginator import EmptyPage
from django.core.paginator import PageNotAnInteger

from django.contrib.auth.decorators import login_required

from django.http import HttpResponseBadRequest

from django.core.files.uploadedfile import SimpleUploadedFile


from .forms import PostForm
from .forms import CommentForm

from .models import Post
from .models import Tag
from .models import Comment

import base64


def index(request):
    per_page = 3
    page = request.GET.get('page', 1) ##???
    posts = Post.objects.all().order_by('-created_at', '-pk')
    pg = Paginator(posts, per_page)
    try:
        contents = pg.page(page)
    except PageNotAnInteger:
        contents = pg.page(1)
    except EmptyPage:
        contents=[]
    ctx={
        'posts':contents,
    }
    return render(request, "ContactDir/index.html", ctx)


def create_post(request):
    if request.method == "GET":
        form = PostForm()
    elif request.method == "POST":
        filtered = request.POST.get('filtered_image')
        if filtered:
            filtered_image = get_base64_image(filtered)
            filename = request.FILES['image'].name.split(os.sep)[-1]
            _filedata = {
                'image' : SimpleUploadedFile(
                    filename, filtered_image
                )
            }
        else:
            _filedata = request.FILES

        form = PostForm(request.POST, _filedata)
        if form.is_valid():
            post = form.save(commit=False)
            post.user = request.user
            post.save()

            tag_text = form.cleaned_data.get('tagtext','') ##???
            tags = tag_text.split(',')
            for _tag in tags:
                _tag = _tag.strip()
                tag,_ = Tag.objects.get_or_create(name=_tag, defaults={'name':_tag})
                post.tags.add(tag)
            return redirect('contact:view', pk=post.pk)
    ctx={
        'form':form,
    }
    return render(request, 'ContactDir/edit_post.html', ctx)


def list_post(request):
    per_page = 3
    page = request.GET.get('page', 1) ##???
    posts = Post.objects.all().order_by('-created_at', '-pk')
    pg = Paginator(posts, per_page)
    try:
        contents = pg.page(page)
    except PageNotAnInteger:
        contents = pg.page(1)
    except EmptyPage:
        contents=[]

    ctx={
        'posts':contents,
    }
    return render(request, 'ContactDir/index.html', ctx)


def view_post(request, pk):
    post = Post.objects.get(pk=pk)

    if request.method == 'GET':
        form = CommentForm(request.GET)
    elif request.method == 'POST':
        form = CommentForm(request.POST)

        if form.is_valid():
            comment = form.save(commit=False)
            comment.user = request.user
            comment.post = post
            comment.save()
            return redirect(post) ## get_absolute_url call
    ctx = {
        'post':post,
        'comment_form':form,
    }
    return render(request, 'ContactDir/view_post.html', ctx)


def delete_comment(request, pk):
    if request.method != 'POST':
        raise HttpResponseBadRequest
    comment = get_object_or_404(Comment, pk=pk)
    comment.delete()
    return redirect(comment.post)##??


def like_post(request, post_pk):
    post = get_object_or_404(Post, pk=post_pk)
    if request.method == 'POST':
        raise Exception('Bad Request')
    qs = post.like_set.filter(user=request.uesr)


def get_base64_image(data):
    if data is None or ';base64' not in data:
        return None

    _format, _content = data.split(';base64,')
    return base64.b64decode(_content)