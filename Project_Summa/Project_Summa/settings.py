"""
Django settings for Project_Summa project.

Generated by 'django-admin startproject' using Django 1.10.4.

For more information on this file, see
https://docs.djangoproject.com/en/1.10/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/1.10/ref/settings/
"""

import os
import raven

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/1.10/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = '+r-2d%)khql&l@4t&#*av(x^zge-qoo29k4r-8m3m)@nh644#5'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True
ALLOWED_HOSTS = []

# DEBUG = False
# ALLOWED_HOSTS = '*'

# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'Home',
    'Analyzer',
    'ProjectManager',
    'Contact',
    'UserProfile',
    'SummaMLEngine',
    'bootstrap3',
    'django_extensions',
    'testapp',
    # 'raven.contrib.django.raven_compat',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    # 'Project_Summa.sample_exceptions.HelloWorldError',
    'Project_Summa.sample_middlewares.SampleMiddleware'
]

ROOT_URLCONF = 'Project_Summa.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'Project_Summa.wsgi.application'

# Database
# https://docs.djangoproject.com/en/1.10/ref/settings/#databases

DATABASES = {
    # 'default': {
    #     'ENGINE': 'django.db.backends.sqlite3',
    #     'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    # },
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'project_summa',
        'USER': 'joo',
        'PASSWORD': '1234', # DB user password
        'HOST': 'localhost',
        'PORT': '',
    }
}


# Password validation
# https://docs.djangoproject.com/en/1.10/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

GRAPH_MODELS = {
    'all_applications': True,
    'group_models': True,
}


##########################################
##
## LOGGING
##
##########################################

# LOGGING = {
#     'version': 1,
#     'disable_existing_loggers': False,
#     'root': {
#         'level': 'DEBUG',
#         'handlers': ['sentry'],
#     },
#     'formatters': {
#         'verbose': {
#             'format': '%(levelname)s %(asctime)s %(module)s %(process)d %(thread)d %(message)s'
#         },
#         'simple': {
#             'format': '%(levelname)s %(message)s'
#         },
#     },
#     'filters': {
#         # 'special': {
#         #     '()': 'project.logging.SpecialFilter',
#         #     'foo': 'bar',
#         # },
#         'require_debug_true': {
#             '()': 'django.utils.log.RequireDebugTrue',
#         },
#     },
#     'handlers': {
#         'console': {
#             'level': 'INFO',
#             'filters': ['require_debug_true'],
#             'class': 'logging.StreamHandler',
#             # 'formatter': 'simple'
#             'formatter': 'verbose'
#         },
#         'sentry': {
#             'level': 'DEBUG',
#             'class': 'raven.contrib.django.handlers.SentryHandler',
#             'formatter': 'verbose',
#         },
#         # 'mail_admins': {
#         #     'level': 'ERROR',
#         #     'class': 'django.utils.log.AdminEmailHandler',
#         #     'filters': ['special']
#         # }
#     },
#     'loggers': {
#         'django': {
#             'handlers': ['console'],
#             'propagate': True,
#         },
#         'raven': {
#             'level': 'DEBUG',
#             'handlers': ['console'],
#             'propagate': False,
#         },
#         'sentry.errors': {
#             'level': 'DEBUG',
#             'handlers': ['console'],
#             'propagate': False,
#         }
#         # 'django.request': {
#         #     'handlers': ['mail_admins'],
#         #     'level': 'ERROR',
#         #     'propagate': False,
#         #  },
#         # 'myproject.custom': {
#         #     'handlers': ['console', 'mail_admins'],
#         #     'level': 'INFO',
#         #     'filters': ['special']
#         # }
#     }
# }
#
# RAVEN_CONFIG = {
#     'dsn': 'https://f54f2341b11041a8a6291be47b6ac223:6321e4925b9545fa82d29bf2b4b0e06e@sentry.io/170019',
#     # If you are using git, you can also automatically configure the
#     # release based on the git info.
#     #'release': raven.fetch_git_sha(os.path.dirname(os.pardir)),
# }

# Internationalization
# https://docs.djangoproject.com/en/1.10/topics/i18n/

LANGUAGE_CODE = 'ko'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/1.10/howto/static-files/

STATIC_URL = '/static/'
STATICFILES_DIRS=[
    os.path.join(BASE_DIR),
    os.path.join(BASE_DIR, "static/"),
]

MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media/')


# http://stackoverflow.com/questions/2882490/how-to-get-the-current-url-within-a-django-template
# for use of current page in template in ProjectManager/views.py
# deprecation warning (1_8.W001) The standalone TEMPLATE_* settings were deprecated in Django 1.8 and the TEMPLATES dictionary takes precedence. You must put the values of the following settings into your default TEMPLATES dict: TEMPLATE_CONTEXT_PROCESSORS.

#TEMPLATE_CONTEXT_PROCESSORS = ('django.core.context_processors.request',)

# TODO: change it to history(-1) page
LOGIN_REDIRECT_URL = ''
LOGOUT_REDIRECT_URL = ''
