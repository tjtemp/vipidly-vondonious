from django.db import models

from django.conf import settings

from django.contrib.auth.models import User

from django.contrib.auth.decorators import login_required
#from UserProfile.models import UserFile

class Analyzer_core(models.Model):
    analyzer_name = models.CharField(max_length=50)

    class Meta:
        ordering=('-pk',)


class Job(models.Model):
    job_status = models.CharField(max_length=30,
                                   choices=((u'pending', u'pending'),
                                            (u'running', u'running'),
                                            (u'finished', u'finished'),
                                            ), default='pending'
                                   )
    job_priority = models.IntegerField(default=1)
    job_name = models.CharField(max_length=50)
    job_submitter = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True)
    job_dataset_size = models.IntegerField(default=0) ## TODO: will be changed
    job_started_at = models.DateTimeField(auto_now=True)
    job_deadline = models.DateTimeField(null=True, blank=True)
    job_completed_at = models.DateTimeField(null=True, blank=True)
    job_solver = models.CharField(max_length=30,
                                  choices=((u'Testing', u'Testing'),
                                            (u'DeepDream', u'DeepDream'),
                                            (u'StyleTransfer', u'StyleTransfer'),
                                            (u'mlsolver', u'mlsolver'),
                                           ), default='Testing'
                                  )

    class Meta:
        ordering = ('pk',)

    def __str__(self):
        return '{}-{} : {}'.format(self.job_name, self.job_submitter, self.job_status)


class Controller(models.Model):
    controller_name = models.CharField(max_length=40, blank=True)
    controller_problem_settings = models.CharField(max_length=40,
                                                  default='classification',
                                                  choices=(
                                                      (u'classification', u'classification'),
                                                      (u'regression',u'regression'),
                                                      (u'clustering', u'clustering'),
                                                    ),
                                                  )
    controller_core_packages = models.CharField(max_length=50,
                                                default='scikit-learn',
                                                choices=(
                                                    (u'sklearn', u'sklearn'),
                                                    (u'lasagne', u'lasagne'),
                                                    (u'keras', u'keras'),
                                                    (u'theano', u'theano'),
                                                    (u'tensorflow', u'tensorflow'),
                                                    (u'torch', u'torch'),
                                                    (u'caffe', u'caffe'),
                                                ),
                                                blank=True,
                                                )
    controller_core_estimator = models.CharField(max_length=50,
                                                     default='LogisticRegression',
                                                 choices=(
                                                     (u'LogisticRegression', u'LogisticRegression'),
                                                     (u'SVM', u'SVM'),
                                                     (u'XgBoost', u'XgBoost'),
                                                     (u'AlexNet', u'AlexNet'),
                                                     (u'GoogleNet', u'GoogleNet'),
                                                     (u'ResNet', u'ResNet'),
                                                          ),
                                                 blank=True,)
    controller_core_parameters = models.CharField(max_length=100, blank=True,)
    controller_visual_tool = models.CharField(max_length=100,
                                              default='scatter',
                                              choices=(
                                                  (u'scatter', u'scatter'),
                                                       )
                                              ,)
    controller_input_file_field = models.FileField(null=True, blank=True)
    controller_created_at = models.DateTimeField(auto_now=True,)

    class Meta:
        ordering=('-pk',)


class PreProcessor(models.Model):
    preprocessor_name = models.CharField(max_length=40, default='test_preprocess')


class PostProcessor(models.Model):
    postprocessor_name = models.CharField(max_length=40, default='test_postprocess')

#from .views import get_user_uploadfile_folder
def get_user_uploadfile_folder(instance, filename):
    '''
    :param upload_apps: classname such as Projects or Task..
    :param filename:
    :return:
    NOTE: FileField has builtin function ._get_path()
    '''
    print("get_user_uploadfile_folder is called")
    return "%s/files/%s" % (instance.fileowner, instance.filename)
    #if isinstance(instance, Image): # User file on user folders
        #return "%s/%s/%s" % (fileclass.image_owner.username, 'image', filename)
    # elif isinstance(fileclass, Video):
    #     return "%s/%s/%s" % (fileclass.image_owner.username, 'video', filename)

class Datafile(models.Model):
    filename = models.CharField(max_length=100, null=True, blank=True)
    filepath = models.FileField(upload_to=get_user_uploadfile_folder, max_length=500,
                                null=True, blank=True)
    fileowner = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True)

#
# class UserAnalyzeFile(models.Model):
#     '''
#     meta data file for user uploaded file
#     '''
#     file_name = models.CharField(max_length=50, default='image_')
#     #file_file = models.FileField(upload_to=get_user_uploadfile_folder, blank=False, null=False)
#     file_owner = models.ForeignKey(settings.AUTH_USER_MODEL)
#
#     class Meta:
#         abstract = True
#

class Image(models.Model):
    image_type = models.CharField(max_length=20, default='image_')
    def __str__(self):
        return '{}'.format(self.image_type)


class Video(models.Model):
    video_type = models.CharField(max_length=20, default='video_')
    def __str__(self):
        return '{}'.format(self.file_name)
    class Meta:
        pass