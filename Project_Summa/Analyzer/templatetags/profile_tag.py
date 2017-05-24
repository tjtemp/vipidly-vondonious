from django import template

from decimal import Decimal

register = template.Library()

@register.filter(name='check_profile_photo_bool')
def check_profile_photo_bool(profile_photo_field):
    return bool(profile_photo_field)