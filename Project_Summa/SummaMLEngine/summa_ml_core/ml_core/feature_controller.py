from .pipeline_controller import *
from .pipeline_builder import *
from .visualization_toolbox import *
from .data_holder import *

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import pandas as pd
import numpy as np
import os
import glob

from scipy import stats


import warnings

class feature_controller(object):
    """ feature control class
    Parameters
    ----------
    feature_selection_option : list,
    X : pandas.DataFrame, pandas.Series
    y : pandas.DataFrame, pandas.Series
    feature_selection_global : list, mask for selected features.

    """
    def __init__(self, X=None, y=None):
        self.feature_selection_option=[]
        self.X = X
        self.y = y
        self.cat_cols, self.con_cols, self.colnames = None, None, None

        if self.X is not None and self.y is not None:
            self.colnames = list(X.columns)
            self.cat_cols, self.con_cols = self.categorize_data(self,X)
        else:
            warnings.warn("You should put data explicitly into "+
                "staticmethod without the declaration of the class.", UserWarning)
        self.feature_selection_global=[]
        return None


    @staticmethod
    def feature_normalization(self, X, y):
        print("normal_distribution_test is conducted..")
        if type(X).__name__ == 'DataFrame':
            for idx, colname in enumerate(X.columns):
                print(X.columns[idx])
                print('t-value : ', stats.shapiro(X[colname])[0])
                print('p-value : ', stats.shapiro(X[colname])[1])
                if stats.shapiro(X[colname])[1] > 0.05 :
                    print("The null hypothesis is disproved.")
        else:
            raise ValueError("Input type should be pandas.DataFrame")

        return (X - X.mean()) / (X.max() - X.min())


    @staticmethod
    def categorize_data(self, data):
        print("categorize_data call!")
        cat_cols, con_cols = [], []
        for cols in list(data.columns):
            if data[cols].dtype == 'object':
                cat_cols.append(cols)
            else:
                con_cols.append(cols)
        return cat_cols, con_cols


    @staticmethod
    def cat_val(self, X, col=None):
        """
        return
        ------
        True : if X[col] datatype is categorical.
        False : if X[col] datatype is continuous.
        """
        if col == None:
            return X.dtype == 'object'
        #print(X[col].dtype)
        return X[col].dtype == 'object'


    @staticmethod
    def cat_val_group(self, X, groupcol=None):
        if groupcol == None:
            return len(np.unique(X))
        return len(np.unique(X[groupcol]))


    @staticmethod
    def norm_test(self, X, col=None):
        """ Check normal distribution test with 95% certainty.
        Parameter
        --------
        X : pandas.DataFrame, pandas.Series, only dtype int or float
        """
        print("norm test call!")
        X = pd.DataFrame(X[col])

        # sanity check
        cat_cols, con_cols = self.categorize_data(self, X)
        print("cat_col : ", cat_cols, "con_cols : ", con_cols)
        if cat_cols:
            raise ValueError("You should put X with only continuous variables.")

        if col == None:
            if len(con_col) == 1:
                return stats.mstats.normaltset(X, axis=0)[1] / 2. > 0.05
            else:
                print("Input has no continuous features.")
                raise ValueError("Input has no continuous features.")
        """
        for idx, col in enumerate(con_cols):
            result = stats.mstats.normaltest(X,
                 axis=0)[1][idx] / 2. > 0.05
#				 axis=X.columns.get_loc(col))[1] / 2. > 0.05
            results.append(result)
        """
        # Single Column input with designated col
        if isinstance(col, str):
            return (stats.normaltest(X.astype("float64"), axis=X.columns.get_loc(col))[1] / 2. <= 0.05).tolist()[0]
        else:
            return (stats.normaltest(X.astype("float64"), axis=X.columns.get_loc(col))[1] / 2. <= 0.05).tolist()


    def param_assumption_test():
        return None


    @staticmethod
    def feature_selection_statistics_auto(self, X, y, groupcol):
        resultset=[]
        print("\tOn group columns '" + groupcol+ "'...")
        print("num of uniq groups in groupcol : ", self.cat_val_group(self, X[groupcol]))
        print("num of uniq groups in groupcol : ", type(self.cat_val_group(self, X[groupcol])))
        for col in self.colnames:
            if col in self.con_cols:
                print("="*50)
                print("on " + col + " columns..")
                if "auto" not in self.feature_selection_option:
                    print("selection options are ..")
                    print(self.feature_selection_option)
                    if "one_way_ANOVA" in self.feature_selection_option:
                        print("one_way_ANOVA is conducted..")
                else:
                    #### single continuous dependent variable problem
                    if self.cat_val(self, X[col]) == False:
                        print("Continuous Variables..")
                        ### univariate independent feature
                        #if len(colnames) == 1:
                        ## categorical independent variable
                        #if col in cat_cols:
                        # 2 groups
                        if self.cat_val_group(self, X[groupcol]) == 2:
                            uniq1 = np.unique(X[groupcol])[0]
                            uniq2 = np.unique(X[groupcol])[1]
                            print("group num == 2 with", uniq1, uniq2)

                            # paired feature
                            if 'paired' in self.feature_selection_option :
                                print('='*50)
                                print('paried test...')
                                if self.norm_test(self, X[self.con_cols], col) == True:
                                    print("paired_t_test is conducted..")
                                    aws = stats.ttest_ind(X[X[groupcol] == uniq1][col],
                                         X[X[groupcol] == uniq2][col])[1] / 2. > 0.05
                                    print(aws)
                                    resultset.append(aws)
                                else:
                                    print("wilcoxon test is conducted..")
                                    aws = stats.wilcoxon(X[X[groupcol] == uniq1][col],
                                        X[X[groupcol] == uniq2][col])[1] / 2. > 0.05
                                    print(aws)
                                    resultset.append(aws)
                            # not paired feature
                            else:
                                if self.norm_test(self, X[self.con_cols], col) == True:
                                    print("two_sample_t_test is conducted..")

                                    aws = stats.ttest_ind(X[X[groupcol] == uniq1][col],
                                        X[X[groupcol] == uniq2][col])[1] / 2. > 0.05
                                    print(aws)
                                    resultset.append(aws)
                                else:
                                    print("Mann_Whitneyu test is conducted..")
                                    aws = stats.mannwhitneyu(X[X[groupcol] == uniq1][col],
                                        X[X[groupcol] == uniq2][col])[1] / 2. > 0.05
                                    print(aws)
                                    resultset.append(aws)
                        # more than 2 groups
                        elif self.cat_val_group(X, groupcol) > 2:
                            uniqs=[]
                            for uniq in np.unique(X[groupcol]):
                                uniqs.append(uniq)
                                if self.norm_test(self, X[self.con_cols], uniq) == True:
                                    print("one_way_ANOVA is conducted on " + col)
                                    aws = stats.f_oneway(X[uniqs])[1]
                                    print(aws)
                                    resultset.append(aws)
                                else:
                                    print("Kruskal-Wallis is conducted on " + col)
                                    aws = stats.kruskal(X[uniqs])[1]
                                    print(aws)
                                    resultset.append(aws)

            elif col in self.cat_cols:
                print("="*50)
                print("on " + col + " columns..")
                print("Categorical variables..")
                print("Ksqr test is conducted on ")
                aws = "catval"
                print(aws)
                resultset.append(aws)
        self.feature_selection_global=resultset
        return None


    def feature_selection_statistics(self, X=None, y=None, selection_option=["auto"]):
        """ feature selection from

        Parameters
        ----------
        X : pandas.DataFrame,
        y : pandas.DataFrame or numpy.array,

        selection_option : list, default = ["auto"]
            ex)
            #### single continuous dependent variable problem
            ### univariate independent feature
            ## categorical independent variable
            # 2 group
            0. paired_t_test
            0. wilcoxon
            0. 2_sample_t_test
            0. Mann_Whitneyu
            # over 3 group
            0. one_way_ANOVA
            0. Kruskal-Wallie
            ## continuous independent variable
            0. Simple Linear Regression

            ### multivariate independent feature
            ## categorical independent variable
            0. two_way_ANOVA
            ## mixed independent variable
            0. GLM
            ## continuous independent variable
            0. Multiple Regression - ols model

            #### Single categorical dependent variable problem
            ### univariate categorical independent feature
            0. Chi2 - SelectKBest

            ### multivariate or continuous independent feature
            0. Logistic Regression

            0. PCA
            0. NMF
            0.

            cf) http://abacus.bates.edu/~ganderso/biology/resources/stats_flow_chart_v2014.pdf

        Return
        ------
        resultset : list, boolean mask of each columns
            ex) [False, Flase, True ...]
        """
        self.feature_selection_option=selection_option
        resultset = {}
        if self.colnames is not None:
            print("all_Cols : ", self.colnames, type(self.colnames).__name__)
        if self.cat_cols is not None:
            print("cat_cols : ", self.cat_cols, type(self.cat_cols).__name__)
        if self.con_cols is not None:
            print("con_cols : ", self.con_cols, type(self.con_cols).__name__)

        #TODO: param assumption test should be changed.
        #TODO: groupcolum loop should result in json format
        #TODO: static method call should get a explicit X, y
        # colum-wise test.
        if X is None and self.X is not None:
            X = self.X
        elif X is None and self.X is None:
            raise ValueError("You should specified at least one of the input X and self.X.")
        if self.cat_cols is None:
            self.cat_cols, _ = self.categorize_data(self,X)
        if len(self.cat_cols) != 0:
            for groupcol in self.cat_cols:
                self.feature_selection_statistics_auto(self, X, y, groupcol)
                resultset[groupcol] = self.feature_selection_global
        print("diff means test selected feature mask : ", resultset)
        return None


    def feature_selection_ml():
        return None


    def feature_selection_model():
        return []

 
