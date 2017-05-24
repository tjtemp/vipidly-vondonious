import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import offsetbox
from time import time

from scipy.stats import randint as sp_randint
from scipy import linalg

# step 0. GET THE DATASET
from sklearn.datasets import (
                load_iris, load_digits, load_boston,
                make_circles, make_classification, make_sparse_spd_matrix,
		        fetch_california_housing
        )

# step 0. BUILD THE PIPELINE HOLDER
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

# step 1. PREPROCESS, DATA TRANFORMATION
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures

# step 2. FEATURE SELECTION, ENGINEERING
from sklearn.feature_selection import (
                f_classif, mutual_info_classif,
                f_regression, mutual_info_regression,
                chi2, SelectKBest, RFE, RFECV,
                SelectFromModel
        )

from sklearn import random_projection
from sklearn import discriminant_analysis
from sklearn.decomposition import (
                PCA,    # Principal Component Analysis
                NMF,     # Non-Negative Matrix Factorization
		FactorAnalysis,
		TruncatedSVD,
        )

from sklearn.manifold import (
		Isomap,
		LocallyLinearEmbedding,
		SpectralEmbedding,
		MDS,
		TSNE,
	)

from sklearn.random_projection import SparseRandomProjection

from sklearn.covariance import GraphLassoCV, ledoit_wolf, LedoitWolf, OAS, ShrunkCovariance, empirical_covariance


# step 3. MODEL SELECTION, TUNING - Hyperparameter Tuning
from sklearn.metrics import (
                classification_report,
                precision_score, recall_score,
                make_scorer,
                zero_one_loss,
                mean_squared_error
        )


from sklearn.model_selection import (
                cross_val_score, cross_val_predict,
                train_test_split, KFold,
                StratifiedKFold, StratifiedShuffleSplit,
                TimeSeriesSplit
        )

# 1. Generalized Linear Model
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.linear_model import LogisticRegression

#
from sklearn.svm import SVC, LinearSVC, SVR

# 10. tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# 11. ensemble
# max_features=n_features for regression, max_features = sqrt(n_features) for classification
from sklearn.ensemble import (
                BaggingClassifier,
                RandomForestClassifier,
                RandomForestRegressor,
                ExtraTreesClassifier,
                ExtraTreesRegressor,
                RandomTreesEmbedding,
                AdaBoostClassifier,
                AdaBoostRegressor,
                GradientBoostingClassifier,
                GradientBoostingRegressor
                )

from sklearn.ensemble.partial_dependence import plot_partial_dependence, partial_dependence

# step 4. PREDICTION with repetition
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


# step 5. VISUALIZATION
from sklearn.model_selection import validation_curve




