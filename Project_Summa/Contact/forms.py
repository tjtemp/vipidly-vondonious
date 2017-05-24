from django import forms

from .models import Post
from .models import Comment

from django.forms import ValidationError


class CommentForm(forms.ModelForm):

    class Meta:
        model = Comment
        fields = ('content',)


class PostForm(forms.ModelForm):
    tagtext = forms.CharField(required=False)

    class Meta:
        model = Post
        fields = ('category', 'title', 'image', 'content', 'tags',)

        widget = {
            'category':forms.TextInput(attrs={'class':'form_control'}),
            'title':forms.TextInput(attrs={'class':'form_control'}),
            # 'image':forms.ClearableFileInput(attrs={'accept':'image/*',
            #                                'onchange':'loadFile(event)'})
        }

    def clean_content(self):
        content = self.cleaned_data['content']
        # banning a word
        if 'fool' in content:
            raise ValidationError('There is an invalid input.')
        return content