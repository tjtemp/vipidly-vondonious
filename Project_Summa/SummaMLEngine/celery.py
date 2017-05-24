from __future__ import absolute_import, unicode_literals
from celery import Celery

app = Celery('SummaMLEngine',
             broker="amqp://guest:guest@localhost:5672/",
             backend='amqp://localhost:5672/',
             include=['SummaMLEngine.task_test'])

# Optional configuration, see the application user guide.
app.conf.update(
    result_expires=3600,
)

if __name__ == '__main__':
    app.start()