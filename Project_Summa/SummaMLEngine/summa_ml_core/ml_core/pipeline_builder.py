from .pipeline_controller import *

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import pandas as pd
import numpy as np
import os
import glob



class pipeline_builder(object):
    """ Pipeline Build helper class

    Parameter
    ---------

    Attribute
    ---------
    param_grid : list, json array format

    """
    def __init__(self, pipeline, params):
        self.pipeline=None
        self.param_grid=None
        self.nested_flag = True

        N_EXPERIMENTS=5
        N_FEATURES_OPTIONS=[4]
        C_OPTIONS=[1,10,100,1000]

        self.non_nested_scores = np.zeros(N_EXPERIMENTS)
        self.nested_scores = np.zeros(N_EXPERIMENTS)

        if params != None:
            self.param_grid = params
        else:
            # default param_grid
            self.param_grid = [
                {
                    'reduce_dim': [PCA(iterated_power=7), NMF()],
                    'reduce_dim__n_components': N_FEATURES_OPTIONS,
                    'classify__C': C_OPTIONS
                },
                {
                    'reduce_dim': [SelectKBest(chi2)],
                    'reduce_dim__k': N_FEATURES_OPTIONS,
                    'classify__C': C_OPTIONS
                },
                ]
        if pipeline != None:
            self.pipeline = pipeline
        else:
            # default pipeline
            self.pipeline = Pipeline([
                ('reduce_dim', PCA()),
                ('classify', SVR()),
                ])


    def random_grid_generator(self, *input_shape):
        """ Random grid points generator with Mersenne Twister
        Paramteter
        ---------
        input_shape : tuple, feature space dimension
        """
        rnd = np.random.RandomState(1)
        rnd.rand(input_shape)


    def param_edit_pop(self):
        self.param_grid.pop()


    def param_edit_add(self,additional_params):
        self.param_grid.append(additional_params)


    def __str__(self):
        return 'Pipeline controller'


    def param_dist_setter(self):
        return {}


    def report_cv(self, results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")


    def build_execute(self, N_EXPERIMENTS, CV_function, X, y, build_option):
        """
        Parameter
        --------
        CV_function : Cross-validation function in sci-kit learn
                Currently only with KFold, StraitifiedKFold is implemented
        """
        start = time()
        if build_option == 1:
            for ith_exp in range(N_EXPERIMENTS):
                # CV technique
                inner_cv = CV_function(n_splits=4, shuffle=True, random_state=ith_exp)
                outer_cv = CV_function(n_splits=4, shuffle=True, random_state=ith_exp)

                clf = GridSearchCV(self.pipeline, param_grid=self.param_grid, n_jobs=-1, cv=inner_cv)
                clf.fit(X, y)
                self.non_nested_scores[ith_exp] = clf.best_score_
                print('On ', ith_exp, 'th experiment GridSearchCV best score : ', clf.best_score_)

                self.nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv)
                self.nested_scores[ith_exp] = self.nested_score.mean()

            print('non_nested_scores : ', self.non_nested_scores)
            print('nested_scores : ', self.nested_scores)

            print("GridSearchCV took %.2f seconds for %d candidate parameter setting in %d experiments."
                  % (time() - start, len(clf.cv_results_['params']), N_EXPERIMENTS))

            print("report on last experiments .. ")
            self.report_cv(clf.cv_results_)
            print("="*50)

        return None


    def SUC_build_pipeline_CV_KFold(self, N_EXPERIMENTS, X, y):
        return pipeline_builder.build_execute(self, N_EXPERIMENTS, KFold, X, y, 1)


    def SUC_build_pipeline_CV_StraitifiedKFlod(self, N_EXPERIMETNS, X, y):
        return pipeline_builder.build_execute(self, N_EXPERIMENTS, StratifiedKFold, X, y, 1)
