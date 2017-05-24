from django.utils.deprecation import MiddlewareMixin
from django.shortcuts import render

from raven.contrib.django.raven_compat.models import sentry_exception_handler

from Project_Summa.sample_exceptions import HelloWorldError

class SampleMiddleware(MiddlewareMixin):
    def process_request(self, request):
        request.just_say = 'HELLO!'


    def process_exception(self, request, exc):
        sentry_exception_handler(request=request)
        if isinstance(exc, HelloWorldError):
            return render(request, 'error/404.html', {
                'error': exc,
                'status': 404,
            })