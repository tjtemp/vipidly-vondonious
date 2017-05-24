# !/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

from .celery import app
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import mpld3
import json


from SummaMLEngine.summa_ml_core.ml_core.pipeline_controller import *
from SummaMLEngine.summa_ml_core.ml_core.feature_controller import *
from SummaMLEngine.summa_ml_core.ml_core.pipeline_builder import *
from SummaMLEngine.summa_ml_core.ml_core.visualization_toolbox import *
from SummaMLEngine.summa_ml_core.ml_core.data_holder import *


from SummaMLEngine.summa_ml_core.image_ml_core.applications.deepdream_styler import styler_deepdream
from SummaMLEngine.summa_ml_core.image_ml_core.applications.neural_styler import styler_transfer

@app.task
def add(x, y):
    return x + y


@app.task
def mul(x, y):
    return x * y


@app.task
def xsum(numbers):
    return sum(numbers)


@app.task
def plot_ols(x, y):
    """
    =========================================================
    Linear Regression Example
    =========================================================
    This example uses the only the first feature of the `diabetes` dataset, in
    order to illustrate a two-dimensional plot of this regression technique. The
    straight line can be seen in the plot, showing how linear regression attempts
    to draw a straight line that will best minimize the residual sum of squares
    between the observed responses in the dataset, and the responses predicted by
    the linear approximation.

    The coefficients, the residual sum of squares and the variance score are also
    calculated.

    """
    # Code source: Jaques Grobler
    # License: BSD 3 clause

    # Load the diabetes dataset
    diabetes = datasets.load_diabetes()

    # Use only one feature
    diabetes_X = diabetes.data[:, np.newaxis, 2]

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

    fig, ax = plt.subplots()

    # Plot outputs
    ax.scatter(diabetes_X_test, diabetes_y_test, color='black')
    ax.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',
             linewidth=3)

    aws = 'Coefficients: {} \n<br/> Mean squared error: {:.2f} \n<br/> Variance score: {:.2f}'.format(regr.coef_,
                                                                np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2),
                                                                regr.score(diabetes_X_test, diabetes_y_test),
                                                                                           )

    #return json.dumps({"console":aws, "plot":mpld3.fig_to_html(fig)})
    return [aws, mpld3.fig_to_html(fig)]
    #return "test"
    #plt.show()


@app.task
def plot_raw_test(x, y):
    print("plot_raw_test called")
    aws = ''
    fig = ''

    dh1 = data_holder()
    dh1.read_data()

    aws += dh1.print_data_table_info("brain_size.csv")

    vt1 = visualization_toolbox(dh1, 'brain_size.csv')
    #result = vt1.raw_data_plot(['11', '21', '22'])
    result = vt1.raw_data_plot(['22'])

    aws += result[0]
    fig += result[1]

    return [aws, fig]


@app.task
def plot_feature_pair_comparision_test(x, y):
    aws = ''
    fig = ''


    ## feature_pair_comparision
    # split 80/20 train-test
    dh1 = data_holder()
    dh1.read_data()
    vt1 = visualization_toolbox(dh1, 'brain_size.csv')
    cal_housing = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(cal_housing.data,
                                                        cal_housing.target,
                                                        test_size=0.2,
                                                        random_state=1)
    names = cal_housing.feature_names

    print("Training GBRT...")
    clf = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                    learning_rate=0.1, loss='huber',
                                    random_state=1)
    clf.fit(X_train, y_train)
    result = vt1.feature_pair_comparision(X_train, y_train, names, clf)
    aws += result[0]
    fig += result[1]
    fig3d = result[2]
    xlabel = result[3]
    ylabel = result[4]
    zlabel = result[5]

    return [aws, fig, fig3d, xlabel, ylabel, zlabel]


@app.task
def plot_feature_covariance_matrix_plot_test(x, y):
    print("feature covariance matrix plot test call")
    aws = ''
    fig = ''

    dh1 = data_holder()
    dh1.read_data()

    vt1 = visualization_toolbox(dh1, 'brain_size.csv')
    result = vt1.feature_covariance_matrix_plot()
    aws += result[0]
    fig += result[1]

    return [aws, fig]


@app.task
def plot_model_cv_score_compare_plot1d_test():
    print("plot_model_cv_score_compare_plot1d test call")
    aws = ''
    fig = ''

    dh1 = data_holder()
    dh1.read_data()
    vt1 = visualization_toolbox(dh1, 'brain_size.csv')


    # Build a classification task using 3 informative features
    X, y = make_classification(n_samples=1000,
                   n_features=10,
                   n_informative=3,
                   n_redundant=0,
                   n_repeated=0,
                   n_classes=2,
                   random_state=0,
                   shuffle=False)

    rf = RandomForestClassifier()

    param_name_svc="gamma"
    param_range_svc=np.logspace(-6, -1, 5)

    param_name_rf="n_estimators"
    param_range_rf=np.array([1,5,20,50,100,200])

    result = vt1.cv_score_compare_plot1d(X, y, param_name_rf, param_range_rf, rf)

    aws += result[0]
    fig += result[1]

    return [aws, fig]


@app.task
def plot_feature_all_test(x, y):
    """
    plot feature engineering
    :param x:
    :param y:
    :return:
    """
    aws = ''
    fig = ''

    print("plot feature all test call")

    ## plot_raw

    fig +="""
          <div class="x_panel">
            <div class="x_title">
              <h2>Raw data plot</h2>
              <ul class="nav navbar-right panel_toolbox">
                <li><a class="collapse-link"><i class="fa fa-chevron-up"></i></a>
                </li>
                <li class="dropdown">
                  <a href="#" class="dropdown-toggle" data-toggle="dropdown" role="button" aria-expanded="false"><i class="fa fa-wrench"></i></a>
                  <ul class="dropdown-menu" role="menu">
                    <li><a href="#">Settings 1</a>
                    </li>
                    <li><a href="#">Settings 2</a>
                    </li>
                  </ul>
                </li>
                <li><a class="close-link"><i class="fa fa-close"></i></a>
                </li>
              </ul>
              <div class="clearfix"></div>
            </div>
            <div class="x_content">

              <p>This represents ... </p>
          """
    result = plot_raw_test(x, y)
    aws += result[0]
    fig += result[1]

    fig += """
            </div>
          </div>
          """

    ## feature_pair_comparision
    fig += "<h3 class='report-title'>pair partial dependence plot</h3>" \
           "<p>This represents ... </p>"
    result = plot_feature_pair_comparision_test(x, y)
    aws += result[0]
    fig += result[1]
    fig3d = result[2]
    xlabel = result[3]
    ylabel = result[4]
    zlabel = result[5]

    fig += "<h3 class='report-title'>covariance matrix plot</h3>" \
           "<p>This represents ... </p>"
    result = plot_feature_covariance_matrix_plot_test(x, y)
    aws += result[0]
    fig += result[1]

    return [aws, fig , fig3d, xlabel, ylabel, zlabel]


@app.task
def plot_model_all_test():
    """
        plot model tuning
    :return:
    """
    print("plot model all test call")
    aws = ''
    fig = ''

    fig += "<h3 class='report-title'>Score with Cross validation and without</h3>" \
           "<p>This represents ... </p>"
    fig += """<script>alert("hi!")</script>"""


    fig += "<div class='x_panel'>" \
                        "<div class='x_title'>" \
                            "<h2>Score with Cross validation and without</h2>" \
           "<ul class='nav navbar-right panel_toolbox'>" \
           "<li><a class='collapse-link'><i class='fa fa-chevron-up'></i></a>" \
           "</li>" \
           "<li class='dropdown'>" \
           "<a href='#' class='dropdown-toggle' data-toggle='dropdown' role='button' aria-expanded='false'><i class='fa fa-wrench'></i></a>" \
           "<ul class='dropdown-menu' role='menu'>" \
           "<li><a href='#'>Settings 1</a>" \
           "</li>" \
           "<li><a href='#'>Settings 2</a>" \
           "</li>" \
           "</ul>" \
           "</li>" \
           "<li><a class='close-link'><i class='fa fa-close'></i></a>" \
           "</li>" \
           "</ul>" \
           "<div class='clearfix'></div>" \
           "</div>" \
           "<div class='x_content' style='display: block;'>"\

    "<p>Simple table with project listing with progress and editing options</p>"\

    "</div>" \
    "</div>" \
    "<!-- Prediction Results panel end -->"



    result = plot_model_cv_score_compare_plot1d_test()

    aws += result[0]
    fig += result[1]

    return [aws, fig]


@app.task
def plot_report_all_test():
    fig = """
     <!-- Prediction Results panel -->
      <div class='x_panel'>
        <div class='x_title'>
          <h2>Model Tuning</h2>
          <ul class='nav navbar-right panel_toolbox'>
            <li><a class='collapse-link"><i class="fa fa-chevron-up'></i></a>
            </li>
            <li class='dropdown'>
              <a href="#" class="dropdown-toggle" data-toggle="dropdown" role="button" aria-expanded="false"><i class="fa fa-wrench"></i></a>
              <ul class="dropdown-menu" role="menu">
                <li><a href="#">Settings 1</a>
                </li>
                <li><a href="#">Settings 2</a>
                </li>
              </ul>
            </li>
            <li><a class="close-link"><i class="fa fa-close"></i></a>
            </li>
          </ul>
          <div class="clearfix"></div>

        </div>

        <div class="x_content" style="display: block;">
          <p>Simple table with project listing with progress and editing options</p>

        </div>
      </div>
      <!-- Prediction Results panel end -->
    """

    return [fig]

@app.task
def plot_user_input(x, y):
    aws = ''
    fig = ''

    print("User input plot")

    result = [aws, fig]

    dh1 = data_holder()
    dh1.read_data()

    aws += dh1.print_data_table_info("brain_size.csv")

    vt1 = visualization_toolbox(dh1, 'brain_size.csv')
    result = vt1.raw_data_plot(['11','21'])

    aws += result[0]
    fig += result[1]

    return [aws, fig]


@app.task
def styler_function(photo_path, option):
    aws = ''
    fig = ''
    print("path in app.task:", photo_path)
    if option == 'DeepDream':
        result = styler_deepdream(photo_path)
        aws += result[0]
        fig += result[1]
    elif option == 'StyleTransfer':
        result = styler_transfer(photo_path)
        aws += result[0]
        fig += result[1]
    print('selected option : ', option)
    return aws, fig


#stackoverflow.com/questions/8506914/detect-whether-celery-is-available-running
def get_celery_worker_status():
    ERROR_KEY = "ERROR"
    try:
        from celery.task.control import inspect
        insp = inspect()
        d = insp.stats()
        if not d:
            d = { ERROR_KEY: 'No running Celery workers were found.' }
    except IOError as e:
        from errno import errorcode
        msg = "Error connecting to the backend: " + str(e)
        if len(e.args) > 0 and errorcode.get(e.args[0]) == 'ECONNREFUSED':
            msg += ' Check that the RabbitMQ server is running.'
        d = { ERROR_KEY: msg }
    except ImportError as e:
        d = { ERROR_KEY: str(e)}
    return d