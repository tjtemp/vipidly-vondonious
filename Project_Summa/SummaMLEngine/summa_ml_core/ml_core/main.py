from pipeline_controller import *
from feature_controller import *
from pipeline_builder import *
from visualization_toolbox import *
from data_holder import *

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import pandas as pd
import numpy as np
import os
import glob


def main():
    print("test")
    dh1 = data_holder()
    dh1.read_data()
    dh1.print_data_table_info("brain_size.csv")

    vt1 = visualization_toolbox(dh1, 'brain_size.csv')
    vt1.raw_data_plot(['11', '21'])

    iris = load_iris()
    X = iris.data
    y = iris.target

    test = pipeline_builder(None, None)
    test.SUC_build_pipeline_CV_KFold(5, X, y)

    fc = feature_controller()
    fc.feature_selection_statistics(pd.DataFrame(X), y, ["auto"])

    # pair
    cal_housing = fetch_california_housing()

    # split 80/20 train-test
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
    vt1.pair_feature_comparision(X_train, y_train, cal_housing.feature_names, clf)


    # feature_importance_plot
    from sklearn.datasets import make_classification

    # Build a classification task using 3 informative features
    X, y = make_classification(n_samples=1000,
                   n_features=10,
                   n_informative=3,
                   n_redundant=0,
                   n_repeated=0,
                   n_classes=2,
                   random_state=0,
                   shuffle=False)

    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=250,
                      random_state=0)
    forest = RandomForestClassifier(n_estimators=20)
    forest.fit(X,y)
    vt1.feature_importance_plot1d(X,y,forest)


    # Supervised classification Problem
    ########################
    # cv_score_compare_plot1d
    svc = SVC()
    rf = RandomForestClassifier()


    param_name_svc="gamma"
    param_range_svc=np.logspace(-6, -1, 5)

    param_name_rf="n_estimators"
    param_range_rf=np.array([1,5,20,50,100,200])

    vt1.cv_score_compare_plot1d(X, y, param_name_rf, param_range_rf, rf)

    ########################
    # cv_nest_compare_plot1d


    ########################
    # cv_target_estimated_plot1d
    X = np.sort(np.random.rand(30))
    y = np.cos(1.5*np.pi*X) + np.random.randn(30)*0.1

    polynomial_features = PolynomialFeatures(include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                ("linear_regression", linear_regression)])

    vt1.cv_target_estimated_plot1d(X, y, "polynomial_features", "degree", [1,4,15], pipeline)

if __name__ == '__main__':
    main()
