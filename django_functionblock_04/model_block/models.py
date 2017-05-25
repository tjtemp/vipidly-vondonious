from django.db import models

class ManyToOneBefore(models.Model):
    modelfield0 = models.ForeignKey('firstmodel')

class firstmodel(models.Model):
    # If there is no primary key in modelfields,
    # django makes 'id' field automatically.
    is_modelfield = True
    # null : default False
    modelfield0 = models.AutoField(primary_key=True)
    modelfield1_1 = models.IntegerField(default=1)
    modelfield1_2 = models.BigIntegerField(null=True,unique=True, verbose_name='64bit integer')
    modelfield1_3 = models.PositiveIntegerField(null=True,blank=True)
    modelfield1_4 = models.SmallIntegerField(null=True,verbose_name='16bit integer')
    modelfield1_5 = models.FloatField(null=True,choices=((1.1, u'first'),(2.1,u'second'),(3.1,u'third')))
    modelfield1_6 = models.DecimalField(null=True,decimal_places=4, max_digits=10)
    modelfield2 = models.BinaryField(null=True,)
    modelfield3_1 = models.CharField(max_length=100, default='hello')
    modelfield3_2 = models.EmailField(null=True,)
    modelfield3_3 = models.URLField(null=True,)
    modelfield3_4 = models.TextField(null=True,)
    modelfield4_1 = models.FileField(null=True,)
    modelfield4_2 = models.ImageField(null=True,upload_to='%yy/%mm/%dd/', auto_created=True)
    modelfield5_1 = models.DateField(auto_now=True)
    modelfield5_2 = models.DateTimeField(auto_now_add=True)
    modelfield6 = models.GenericIPAddressField(null=True,)
    modelfield7_1 = models.BooleanField(default=True)
    modelfeild7_2 = models.NullBooleanField()

    def __str__(self):
        return '{} - {}'.format(self.modelfield3_1, self.modelfield1_1)

    def get_absolute_url(self):
        pass

    class Meta:
        ordering = ['-modelfield5_1',]
        get_latest_by = 'modelfield5_1'
        #index_together, db_table, managed

class OneToOneModel(models.Model):
    modelfield0 = models.OneToOneField(firstmodel)

class OneToManyModel(models.Model):
    modelfield0 = models.ForeignKey(firstmodel)

class ManyToManyModel(models.Model):
    modelfield0 = models.ManyToManyField(firstmodel)
