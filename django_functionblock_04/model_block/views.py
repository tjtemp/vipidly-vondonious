from django.shortcuts import render

from .models import firstmodel

def index(request):
    f1 = firstmodel()
    f1.modelfield1 = 1
    f1.save()
    f2 = firstmodel()
    f2.modelfield1 = 2
    f2.save()
    for i in range(3,10):
        f = firstmodel()
        f.modelfield1 = i
        f.save()
    # ORM sample
    qs_firstmodel = firstmodel.objects.all()

    # QuerySet method
    qs_fm_get = firstmodel.objects.get(pk=1)
    print("QuerySet get method : ", qs_fm_get)
    print("QuerySet earliest method : ", firstmodel.objects.earliest())

    # QuerySet argument

    ctx = {
        'qs_firstmodel': qs_firstmodel,
    }
    return render(request, 'template_sample.html', ctx)