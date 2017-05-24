from django.forms import ModelForm

from .models import UserProfile
from .models import UserMessage

class UpdateUserProfileForm(ModelForm):
    class Meta:
        model = UserProfile
        fields = '__all__'


class SendUserMessageForm(ModelForm):
    class Meta:
        model = UserMessage
        fields = '__all__'