import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time

from scipy.stats import randint as sp_randint

# step 0. GET THE DATASET
from sklearn.datasets import (
		load_iris, load_digits, load_boston,
		make_circles, make_classification
	)

# step 0. BUILD THE PIPELINE HOLDER
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

# step 1. PREPROCESS, DATA TRANFORMATION
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# step 2. FEATURE SELECTION, ENGINEERING
from sklearn.feature_selection import (
		f_classif, mutual_info_classif,
		f_regression, mutual_info_regression,
		chi2, SelectKBest, RFE, RFECV,
		SelectFromModel
	)

from sklearn.decomposition import (
		PCA,	# Principal Component Analysis
		NMF	# Non-Negative Matrix Factorization

	)

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

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

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
from sklearn.svm import SVC, LinearSVC

# step 4. PREDICTION with repetition
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

################################################
# database-holder, feature controller, Model controller, visualization toolbox, 


#p1 = make_pipeline(StandardScaler(), SVC(C=1))
#cross_val_score(p1, dataset, target, cv=StratifiedShuffleSplit())
#preds = cross_val_predict(p1, dataset, target, cv=10)
#accuracy_score(target, preds)


# running mode flag
rflag = 3


# DataSet mode flag
# dataset, pipeline, visulization
dflag = []

# visualization flag
# '01' : compare with cross-validation predictions and test
#	x : target
#	y : predicted target with CV
# '02' : optimal number of feature with recursive feature elimination
#	x :
#	y :

vflag = ['02', 'nested_score_comp']

if rflag == 1:
	# 1. bare test
	iris = load_iris()
	data = iris.data
	target = iris.target

	Xtrain, Xtest, ytrain, ytest = train_test_split(data, target, test_size=0.2, random_state=0)

	clf = SVC(kernel='rbf', C=0.1)
	clf.fit(Xtrain, ytrain)
	print("Accuracy with single classification : ",clf.score(Xtest, ytest))
	scores = cross_val_score(clf, Xtrain, ytrain ,cv=3) 
	print("Accuracy with cross validation : {0:.3f} +- {1:.3f}".format(scores.mean(), scores.std()*2))

	print("="*30)

	preds = cross_val_predict(clf, Xtrain, ytrain, cv=10)

	print(preds)

	
	if '01' in vflag:
		## compare with cross-validation predictions and test
		fig, ax = plt.subplots()
		ax.scatter(target, preds)
		ax.plot([target.min(), target.max()], [preds.min(), preds.max()], 'k--', lw=4)
		ax.set_xlabel('Measured')
		ax.set_ylabel('Predicted')
		plt.show()

	if '02' in vflag:
		## Check optimal number of feature with recursive feature elimination
		# n_classes*n_cluster_per_class <= 2**n_informative
		X, y = make_classification(n_samples=500, n_features=25, n_informative=3,
					n_redundant=2, n_repeated=0, n_classes=8,
					n_clusters_per_class=1, random_state=0)
		svc = SVC(kernel="linear")
		rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2), scoring='accuracy')
		rfecv.fit(X, y)
		
		plt.figure()
		plt.title("Optimal number of features {}".format(rfecv.n_features_))
		plt.xlabel("Number of feature selected")
		plt.ylabel("Cross validation score (nb of correct classification)")
		plt.plot(range(1, len(rfecv.grid_scores_)+1), rfecv.grid_scores_)
		plt.show()


elif rflag == 2:
	# 2. single classifier with CV

	print("#2. Pipelining with CV")
	digits = load_digits()
	n_samples = len(digits.images)
	X = digits.data
	y = digits.target

	Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)

	tuned_parameters = [
		{'kernel': ['rbf'], 'gamma':[1e-3, 1e-4], 'C':[1, 10, 100, 1000]},
		{'kernel': ['linear'], 'C':[1, 10, 100, 1000]}
	]

	scores = ['precision', 'recall']

	for score in scores:
		print("="*50)
		print("for score :", score)
		print("="*50)
		clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring='%s_macro' % score)
		clf.fit(Xtrain, ytrain)
		print("Possible parameter options..\n")
		print(sorted(clf.cv_results_.keys()))
		print("Best parameter set found on development set :\n")
		print(clf.best_params_)
		
		means = clf.cv_results_['mean_test_score']
		stds = clf.cv_results_['std_test_score']
		
		for mean, std, params in zip(means, stds, clf.cv_results_['params']):
			print(" {0:.3f} +/- {1:.3f} for {2}".format(mean, std * 2, params))

		print()
		
		y_true, y_pred = ytest, clf.predict(Xtest)
		print(classification_report(y_true, y_pred))


elif rflag == 3:
	# 3. Pipelining with CV
	print("# 3. Pipling with CV and multiple experiments")

	def report(results, n_top=3):
		for i in range(1, n_top + 1):
			candidates = np.flatnonzero(results['rank_test_score'] == i)
			for candidate in candidates:
				print("Model with rank: {0}".format(i))
				print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
				results['mean_test_score'][candidate],
				results['std_test_score'][candidate]))
				print("Parameters: {0}".format(results['params'][candidate]))
				print("")
	
	"""
	iris = load_iris()
	X = iris.data
	y = iris.target
	"""
	# NMF ERROR!
	X, y = make_classification(n_samples=100, n_features=10, n_informative=3,
					n_redundant=2, n_repeated=0, n_classes=8, 
					n_clusters_per_class=1, random_state=0)
	X = X + [50]*10

	pipe = Pipeline([
	    ('reduce_dim', PCA()),
	    ('classify', LinearSVC())
	])


	N_EXPERIMENTS = 5
	N_FEATURES_OPTIONS = [4]
	C_OPTIONS = [1, 10, 100, 1000]

	reducer_labels = ['PCA', 'NMF', 'KBest(chi2)']


	non_nested_scores = np.zeros(N_EXPERIMENTS)
	nested_scores = np.zeros(N_EXPERIMENTS)

	############################################################

	param_grid = [
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

	print('Grid Search experiments... ')
	start = time()
	for ith_exp in range(N_EXPERIMENTS):

		# CV technique
		inner_cv = KFold(n_splits=4, shuffle=True, random_state=ith_exp)
		outer_cv = KFold(n_splits=4, shuffle=True, random_state=ith_exp)

		clf = GridSearchCV(pipe, param_grid=param_grid, n_jobs=-1, cv=inner_cv)
		clf.fit(X, y)
		non_nested_scores[ith_exp] = clf.best_score_
		print('On ', ith_exp, 'th experiment GridSearchCV best score : ', clf.best_score_)
		
		nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv)
		nested_scores[ith_exp] = nested_score.mean()
		
	print('non_nested_scores : ', non_nested_scores)
	print('nested_scores : ',nested_scores)

	print("GridSearchCV took %.2f seconds for %d candidate parameter setting in %d experiments."
	      % (time() - start, len(clf.cv_results_['params']), N_EXPERIMENTS))

	
	print("report on last experiments .. ")
	report(clf.cv_results_)
	print("="*50)


	############################################################

	# specify parameters and distributions to sample from
	param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 8), # should be down to 2 when pipe used
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

	# run randomized search
	n_iter_search = 20

	### TODO:
	# estimator should contain bootstrap as param, check estimator.get_params()
	pipe = RandomForestClassifier(n_estimators=20)
	##
	print('Randomized Search experiments... ')
	start = time()
	for ith_exp in range(N_EXPERIMENTS):

		# CV technique
		inner_cv = KFold(n_splits=4, shuffle=True, random_state=ith_exp)
		outer_cv = KFold(n_splits=4, shuffle=True, random_state=ith_exp)

		clf = RandomizedSearchCV(pipe, param_distributions=param_dist,
				 n_jobs=-1, cv=inner_cv, n_iter=n_iter_search)
		clf.fit(X, y)
		non_nested_scores[ith_exp] = clf.best_score_
		print('On ', ith_exp, 'th experiment GridSearchCV best score : ', clf.best_score_)
		
		nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv)
		nested_scores[ith_exp] = nested_score.mean()
		
	print('non_nested_scores : ', non_nested_scores)
	print('nested_scores : ',nested_scores)

	print("RandomizedSearchCV took %.2f seconds for %d candidates"
	      " parameter settings in %d experiments." % ((time() - start), n_iter_search,
								N_EXPERIMENTS))

	print("report on last experiments .. ")
	report(clf.cv_results_)
	print("="*50)


	if 'nested_score_comp' in vflag:

		score_difference = non_nested_scores - nested_scores

		print("Average difference of {0:6f} with std. dev. of {1:6f}."
		      .format(score_difference.mean(), score_difference.std()))

		# Plot scores on each trial for nested and non-nested CV
		plt.figure()
		plt.subplot(211)
		non_nested_scores_line, = plt.plot(non_nested_scores, color='r')
		nested_line, = plt.plot(nested_scores, color='b')
		plt.ylabel("score", fontsize="14")
		plt.legend([non_nested_scores_line, nested_line],
			   ["Non-Nested CV", "Nested CV"],
			   bbox_to_anchor=(0, .4, .5, 0))
		plt.title("Non-Nested and Nested Cross Validation on Iris Dataset",
			  x=.5, y=1.1, fontsize="15")

		# Plot bar chart of the difference.
		plt.subplot(212)
		difference_plot = plt.bar(range(N_EXPERIMENTS), score_difference)
		plt.xlabel("Individual Trial #")
		plt.legend([difference_plot],
			   ["Non-Nested CV - Nested CV Score"],
			   bbox_to_anchor=(0, 1, .8, 0))
		plt.ylabel("score difference", fontsize="14")

		plt.show()
