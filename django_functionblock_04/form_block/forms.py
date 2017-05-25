from django import forms

from .models import SampleModel
from .models import ImageModel
from django.utils.translation import ugettext_lazy as _
#
# django form attributes append
# 0. models.py 1. forms.py 2. views
# In Html code
# Q. how to access for formfield
# Q. How to access attribute and value for formfield

# decorator
from functools import partial
DateInput = partial(forms.DateInput, {'class': 'datepicker'})

class SampleForm(forms.Form):
    #  formField gets arguments as default widget takes arguments
    formfield_1 = forms.BooleanField(required=False, initial=True, label='firstly')
    formfield_2_1 = forms.CharField(widget=forms.TextInput(attrs={'class':'classattr_css'}))
    formfield_2_2 = forms.CharField(initial='name')#, attrs={'class':'classattr_css'}
    formfield_2_3 = forms.CharField(widget=forms.Textarea, max_length=100, help_text='same with initial')
    formfield_2_4 = forms.CharField(widget=forms.PasswordInput(attrs={'class':'trial'})) # initial not working, passwordIput extends TextInput
    formfield_2_5 = forms.CharField(widget=forms.HiddenInput)
    formfield_2_6 = forms.CharField(widget=forms.CheckboxInput)
    formfield_2_7 = forms.CharField(widget=forms.Select(choices=((1,'one'),(2,'two'))))
    formfield_2_8 = forms.CharField(widget=forms.RadioSelect(attrs={'class': 'hello'}, choices=((1,'first'),(2,'second'),(3,'third')))) # default Select()
    formfield_2_2_1 = forms.CharField(error_messages={'required':'your name! please!'}) # when required is failed
    formfield_2_2_2 = forms.TextInput(attrs={'title':'txtinput'})
    formfield_2_2_2.render('nametest','valuetest') # name and value

    formfield_3 = forms.ChoiceField(choices=((1,'one'),(2,'two')))

    formfield_4_1 = forms.DateField(input_formats=['%Y-%m-%d',],initial='2016-12-1', widget=forms.TextInput(attrs={"onfocus":"return this.value=''"}))
    formfield_4_2 = forms.DateField(widget=DateInput())
    formfield_5 = forms.BooleanField(required=False, initial=True)
    formfield_6 = forms.EmailField(widget=forms.TextInput(attrs={'class':'classattr_css'}))
    formfield_7 = forms.FileField(widget=forms.FileInput(attrs={'class':'btn btn-primary bn-file'}))
    formfield_7_2 = forms.FileField()
    formfield_7_3 = forms.ImageField(widget=forms.TextInput(attrs={'class':'btn btn-primary'}))
    #formfield_8 = forms.FilePathField(choices=((1,'a'),(2,'b')))
    # SampleForm(auto_id=False,initial={'formfield_1':'1'},prefix='tsmp')



class SampleModelForm(forms.ModelForm):
    class Meta:
        model = SampleModel
        #fields = ('firstfield',)
        fields = '__all__'
        error_messages = {}
        widgets = {
            'modelfield_1': forms.Textarea(attrs={'cols': 80, 'rows': 20}),
        }
        labels = {
            'modelfield_1': _('Writer'),
        }
        help_texts = {
            'modelfield_1': _('Some useful help text.'),
        }
        error_messages = {
            'modelfield_1': {
                'max_length': _("This writer's name is too long."),
            },
        }

class ImageModelForm(forms.ModelForm):
    class Meta:
        model = ImageModel
        fields = ('file', )