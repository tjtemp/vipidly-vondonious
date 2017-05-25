from django.db import models


# models field type  -  corresponding form field
# Model field	Form field
# AutoField	Not represented in the form
# BigAutoField	Not represented in the form
# BigIntegerField	IntegerField with min_value set to -9223372036854775808 and max_value set to 9223372036854775807.
# BooleanField	BooleanField
# CharField	CharField with max_length set to the model fields max_length
# CommaSeparatedIntegerField	CharField
# DateField	DateField
# DateTimeField	DateTimeField
# DecimalField	DecimalField
# EmailField	EmailField
# FileField	FileField
# FilePathField	FilePathField
# FloatField	FloatField
# ForeignKey	ModelChoiceField (see below)
# ImageField	ImageField
# IntegerField	IntegerField
# IPAddressField	IPAddressField
# GenericIPAddressField	GenericIPAddressField
# ManyToManyField	ModelMultipleChoiceField (see below)
# NullBooleanField	NullBooleanField
# PositiveIntegerField	IntegerField
# PositiveSmallIntegerField	IntegerField
# SlugField	SlugField
# SmallIntegerField	IntegerField
# TextField	CharField with widget=forms.Textarea
# TimeField	TimeField
# URLField	URLField

class SampleModel(models.Model):
    modelfield_1 = models.CharField(max_length=100)
    modelfield_2 = models.IntegerField(null=True, blank=True)
    modelfield_3 = models.BooleanField(default=False)
    modelfield_4_1 = models.DateField(auto_now_add=True)
    modelfield_4_2 = models.DateField(null=True, blank=True)
    modelfield_5 = models.DecimalField(decimal_places=5,max_digits=5,null=True, blank=True)
    modelfield_6 = models.FileField(null=True, blank=True)
    modelfield_7 = models.EmailField(null=True, blank=True)

class ImageModel(models.Model):
    title = models.CharField(max_length=255, blank=True)
    file = models.FileField(upload_to='photos/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
