from django.conf.urls import url

from .views import index
from .views import ml_core
from .views import ksprp # kaggle-santander-product-recommendation-problem
from .views import ktncfmp

#from .views import image_core
from .views import ksfddp # kaggle-state-farm-distracted-driver-detectection-problem
from .views import ktsfmcp

from .views import kocpp # kaggle-outbrain-click-prediction-problem

from .views import nlp_ml_core
from .views import kqqpp # kaggle-quora-question-pairs-problem

from .views import video_analysis
from .views import image_ml_core
from .views import image_preprocess
from .views import FileFieldView

app_name = 'analyzer'

urlpatterns=[
    #/analyzer/
    url(r'^$', index ,name="index"),
    url(r'^image-analysis/$', image_ml_core, name="image-ml-core"),
    url(r'^image-analysis/image-preprocess/$', image_preprocess, name="image-preprocess"),
    url(r'^image-analysis/multiple-image-upload-test/$', FileFieldView.as_view(), name="multiple-image-upload-test"),
    url(r'^ml-core/$', ml_core, name="ml-core"),
    url(r'^nlp-core/$', nlp_ml_core, name="nlp-ml-core"),
    url(r'^kaggle-state-farm-distracted-driver-problem/',
        ksfddp , name="kaggle-state-farm-distracted-driver-problem"),
    url(r'^kaggle-santander-product-recommendation-problem/',
        ksprp , name="kaggle-santander-product-recommendation-problem"),
    url(r'^kaggle-the-nature-conservancy-fisheries-monitoring-problem/',
        ktncfmp, name="kaggle-the-nature-conservancy-fisheries-monitoring-problem"),
    url(r'^kaggle-two-sigma-financial-modeling-challenge-problem/',
        ktsfmcp, name='kaggle-two-sigma-financial-modeling-challenge-problem'),
    url(r'kaggle-quora-question-pairs-problem/',
        kqqpp, name='kaggle-quora-question-pairs-problem'),
    url(r'kaggle-outbrain-click-prediction-problem/',
        kocpp, name='kaggle-outbrain-click-prediction-problem')
]