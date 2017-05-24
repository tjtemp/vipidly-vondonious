from .pipeline_controller import *
from .feature_controller import *
from .pipeline_builder import *
from .data_holder import *

import matplotlib.pyplot as plt
import mpld3
from mpld3 import plugins, utils
plt.style.use('ggplot')
import seaborn as sns
import pandas as pd
import numpy as np
import os
import glob
import json

import sklearn.ensemble as ensemble


class visualization_toolbox:
    """
    -Raw_data_plot
      |_Univariate Feature visualization
      |_Pairwise Feature visualization
    -Pre_Analysis_plot
      |_Univariate Feature visualization
      |_Pairwise Feature visualization
    -Post_Analysis_plot
      |_Univariate Feature visualization
      |_Pairwise Feature visualization

      raw_data_plot
      feature_pair_comparision
      feature_covariance_matrix_plot
      feature_importance_plot1d
      feature_importance_plot2d
      feature_selection_plot
      feature_dim_reduction_plot

      :plot issue:
      -Embedding Seaborn plot in WxPython panel
      http://stackoverflow.com/questions/31758391/embedding-seaborn-plot-in-wxpython-panel
      -Plotting with seaborn using the matplotlib object-oriented interface
      http://stackoverflow.com/questions/23969619/plotting-with-seaborn-using-the-matplotlib-object-oriented-interface
      -seaborn produces separate figures in subplots
      http://stackoverflow.com/questions/33925494/seaborn-produces-separate-figures-in-subplots

    """
    def __init__(self, dataholder, table_name):
        #TODO: really need?
        self.dh = dataholder
        self.dh.read_data()
        self.table_name = table_name
        #table_name = list(col_args)
        # Univariate plot
        self.table_index = self.dh.datatable_names.index(str(self.table_name))
        self.table = self.dh.datatables[self.table_index]
        self.native_flag = False


    def change_table(self, table_name):
        self.table_name = table_name
        self.table_index = self.dh.datatable_names.index(str(self.table_name))
        self.table = self.dh.datatables[self.table_index]


    def raw_data_plot(self, plot_option, *col_args):
        """
            input :
                plot_option : index of plot option
                (11=bar plot, 21=two-pair scatterplot, 22=two-pair jointplot)
            return :
                None
        """
        # Raw-data-visualization
        # Univariate plot

        cat_cols, con_cols = self.dh.categorize_data(self.table_name)

        cr = '' # console report
        pr = '' # plot report
        delim = '<br/>'

        if '11' in plot_option:
            for col in con_cols:
                if self.native_flag == True:
                    print('='*50)
                    print('column name : ', col)
                else:
                    cr += '='*50
                    cr += delim
                    cr += 'column name : {}'.format(col)
                    cr += delim

                fig, ax = plt.subplots()
                ax.set_title(col)
                type(self.table)
                sns.countplot(x=col, data=self.table, alpha=0.5)
                if self.native_flag == True:
                    plt.show()
                else:
                    pr += mpld3.fig_to_html(fig) + delim

        if '21' in plot_option:
            if self.native_flag == True:
                print(con_cols)
            else:
                cr += str(con_cols)
            #self.table[con_cols].replace(np.nan, 0, inplace=True)
            fig, ax = plt.subplots()
            sns.pairplot(self.table[con_cols].dropna())
            if self.native_flag == True:
                plt.show()
            else:
                pr += mpld3.fig_to_html(fig) + delim


        import itertools

        if '22' in plot_option:
            for col_pair in itertools.combinations(con_cols, 2):
                if self.native_flag == True:
                    print(col_pair)
                else:
                    cr += str(col_pair)
                fig, ax = plt.subplots()

                ax.set_title(col_pair[0]+" vs. "+col_pair[1])
                sns.jointplot(col_pair[0], col_pair[1], data=self.table, kind="kde", ax=ax)

                pr += mpld3.fig_to_html(fig) + delim

        return [cr, pr]


    # feature_selection, model_selection
    # svm(default), ensemble, neural_network, manifold learning

    # Pre-analysis-visualization

    # Post-analysis-visualization
    # http://scikit-learn.org/stable/auto_examples/ensemble/plot_partial_dependence.html
    def feature_pair_comparision(self, X, y, feature_names, fitted_model):
        """ F-value, mutual Information, for two feature and partial dependence
        """
        cr = ''
        delim = '<br/>'
        #TODO: currently only paritial dependence is adapted.
        f_test = f_regression(X, y)
        f_test /= np.max(f_test)

        mi = mutual_info_regression(X, y)
        mi /= np.max(mi)
        if self.native_flag == True:
            print("f-value : ", f_test)
            print("mutual info : ", mi)

            print('Convenience plot with ``partial_dependence_plots``')
        else:
            cr += "f-value : {}".format(f_test) + delim
            cr += "mutual info : ".format(mi) + delim
            cr += 'Convenience plot with ``partial_dependence_plots``' + delim

        features = [0, 5, 1, 2, (5, 1)]
        fig, axs = plot_partial_dependence(fitted_model, X, features,
                           feature_names=feature_names,
                           n_jobs=3, grid_resolution=50)
        fig.suptitle('Partial dependence of house value on nonlocation features\n'
             'for the California housing dataset')
        plt.subplots_adjust(top=0.9)  # tight_layout causes overlap with suptitle

        if self.native_flag == True:
            print('Custom 3d plot via ``partial_dependence``')
        else:
            cr += 'Custom 3d plot via ``partial_dependence``' + delim
        #fig = plt.figure()

        target_feature = (1, 5)
        pdp, axes = partial_dependence(fitted_model, target_feature,
                       X=X, grid_resolution=50)
        XX, YY = np.meshgrid(axes[0], axes[1])
        Z = pdp[0].reshape(list(map(np.size, axes))).T
        print("Z shape : ", Z.shape)

        if self.native_flag == True:
            ax = Axes3D(fig)
            surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1, cmap=plt.cm.BuPu)
            ax.set_xlabel(feature_names[target_feature[0]])
            ax.set_ylabel(feature_names[target_feature[1]])
            ax.set_zlabel('Partial dependence')
            #  pretty init view
            ax.view_init(elev=22, azim=122)
            plt.colorbar(surf)
            plt.suptitle('Partial dependence plot')
            plt.subplots_adjust(top=0.9)

            plt.show()
        else:
            return [cr, mpld3.fig_to_html(fig), Z.tolist(),
                             feature_names[target_feature[0]],
                             feature_names[target_feature[1]],
                             'partial dependence'
                             ]


    #http://scikit-learn.org/stable/auto_examples/covariance/plot_sparse_cov.html#sphx-glr-auto-examples-covariance-plot-sparse-cov-py
    def feature_covariance_matrix_plot(self, X=None, y=None):
        """ Covariance matrix plot with Empirical(MSE), shrunk, sparse inverse covariance.
        """
        cr = ''
        delim = '<br/>'

        prng = np.random.RandomState(1)

        if X is None or y is None:

            n_samples = 60
            n_features = 20

            prec = make_sparse_spd_matrix(n_features, alpha=.98,
                              smallest_coef=.4,
                              largest_coef=.7,
                              random_state=prng)
        else:
            n_samples = X.shape[0]
            n_features = X.shape[1]
        # Covariance matrix
        cov = linalg.inv(prec)
        d = np.sqrt(np.diag(cov))
        cov /= d
        cov /= d[:, np.newaxis]
        prec *= d
        prec *= d[:, np.newaxis]
        X = prng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)
        X -= X.mean(axis=0)
        X /= X.std(axis=0)

        # Estimate the covariance
        emp_cov = np.dot(X.T, X) / n_samples

        model = GraphLassoCV()
        model.fit(X)
        cov_ = model.covariance_
        prec_ = model.precision_

        lw_cov_, _ = ledoit_wolf(X)
        lw_prec_ = linalg.inv(lw_cov_)



        fig = plt.figure(figsize=(10, 6))
        plt.subplots_adjust(left=0.02, right=0.98)



        # Ledoit-Wolf optimal shrinkage coefficient estimate
        lw = LedoitWolf()
        lw.fit(X)

        # OAS coefficient estimate
        oa = OAS()
        oa.fit(X)



        # plot the covariances
        covs = [('Empirical', emp_cov), ('Ledoit-Wolf', lw_cov_),
            ('GraphLasso', cov_), ('True', cov)]
        vmax = cov_.max()
        for i, (name, this_cov) in enumerate(covs):
            fig, plt.subplot(2, 4, i + 1)
            plt.imshow(this_cov, interpolation='nearest', vmin=-vmax, vmax=vmax,
                   cmap=plt.cm.RdBu_r)
            plt.xticks(())
            plt.yticks(())
            plt.title('%s covariance' % name)


        # plot the precisions
        precs = [('Empirical', linalg.inv(emp_cov)), ('Ledoit-Wolf', lw_prec_),
             ('GraphLasso', prec_), ('True', prec)]
        vmax = .9 * prec_.max()
        for i, (name, this_prec) in enumerate(precs):
            ax = plt.subplot(2, 4, i + 5)
            plt.imshow(np.ma.masked_equal(this_prec, 0),
                   interpolation='nearest', vmin=-vmax, vmax=vmax,
                   cmap=plt.cm.RdBu_r)
            plt.xticks(())
            plt.yticks(())
            plt.title('%s precision' % name)
            ax.set_axis_bgcolor('.7')

        # plot the model selection metric
        """
        plt.figure(figsize=(4, 3))
        plt.suptitle('selected shrinkage coeff(regularization) : {}'.format(model.alpha_), y=1.0)
        plt.axes([.2, .15, .75, .7])
        plt.plot(model.cv_alphas_, np.mean(model.grid_scores, axis=1), 'o-')
        plt.axvline(model.alpha_, color='.5', label='CV')

        # LW likelihood
        plt.axvline(lw.shrinkage_, color='magenta',
               linewidth=3, label='Ledoit-Wolf estimate')
        # OAS likelihood
        plt.axvline(oa.shrinkage_, color='purple',
               linewidth=3, label='OAS estimate')

        plt.title('Model selection')
        plt.ylabel('Cross-validation score')
        plt.xlabel('alpha')
        """
        if self.native_flag == True:
            plt.show()

        else:
            return [cr, mpld3.fig_to_html(fig)]


    # http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#sphx-glr-auto-examples-ensemble-plot-forest-importances-py
    def feature_importance_plot1d(self, X, y, modelclass):
        if hasattr(modelclass, 'feature_importances_'):
            importances = modelclass.feature_importances_
            std=0
            indices=[]
            if type(modelclass).__name__ in dir(ensemble):
                std = np.std([modelclass.feature_importances_ for modelclass in modelclass.estimators_], axis=0)
                indices = np.argsort(importances)[::-1]
            print("Feature ranking")
            for f in range(X.shape[1]):
                print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(X.shape[1]), importances[indices],
               color="r", yerr=std[indices], align="center")
        plt.xticks(range(X.shape[1]), indices)
        plt.xlim([-1, X.shape[1]])
        plt.show()


    def feature_importance_plot2d(self, X, y, modelclass):
        if hasattr(modelclass, 'feature_importances_'):
            importances = modelclass.feature_importances_
            std=0
            indices=[]
            if type(modelclass).__name__ in dir(ensemble):
                std = np.std([modelclass.feature_importances_ for modelclass in modelclass.estimators_], axis=0)
                indices = np.argsort(importances)[::-1]
            print("Feature ranking")
            for f in range(X.shape[1]):
                print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        plt.figure()
        plt.title("Feature importances")
        importances = importances.reshape(int(np.sqrt(X.shape[1])),-1)
        plt.matshow(importances, cmap=plt.cm.hot)
        plt.show()


    def feature_selection_plot(self, X, y, model):
        rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(2), scoring='accuracy')
        rfecv.fit(X, y)

        print("Optimal number of features : %d" % rfecv.n_features_)

        # Plot number of features VS. cross-validation scores
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.show()


    # http://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html#sphx-glr-auto-examples-manifold-plot-lle-digits-py
    # data Structure visualization
    def feature_dim_reduction_plot(self, X, y, imgX=None):
        imgX = load_digits().images
        n_samples, n_features = X.shape
        n_neighbors = 30

        # Scale and visualize the embedding vectors
        def plot_embedding(X, title=None):
            x_min, x_max = np.min(X, 0), np.max(X, 0)
            X = (X - x_min) / (x_max - x_min)

            plt.figure()
            ax = plt.subplot(111)
            for i in range(X.shape[0]):
                plt.text(X[i, 0], X[i, 1], str(y[i]),
                color=plt.cm.Set1(y[i] / 10.),
                fontdict={'weight': 'bold', 'size': 9})

            if hasattr(offsetbox, 'AnnotationBbox') and imgX is not None:
                # only print thumbnails with matplotlib > 1.0
                shown_images = np.array([[1., 1.]])  # just something big
                for i in range(X.shape[0]):
                    dist = np.sum((X[i] - shown_images) ** 2, 1)
                    if np.min(dist) < 4e-3:
                    # don't show points that are too close
                        continue
                    shown_images = np.r_[shown_images, [X[i]]]
                    imagebox = offsetbox.AnnotationBbox(
                        offsetbox.OffsetImage(imgX[i], cmap=plt.cm.gray_r),
                        X[i])
                    ax.add_artist(imagebox)
            plt.xticks([]), plt.yticks([])
            if title is not None:
                plt.title(title)


        #----------------------------------------------------------------------
        # Random 2D projection using a random unitary matrix
        print("Computing random projection")
        rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)
        X_projected = rp.fit_transform(X)
        plot_embedding(X_projected, "Random Projection of the digits")


        #----------------------------------------------------------------------
        # Projection on to the first 2 principal components

        print("Computing PCA projection")
        t0 = time()
        X_pca = TruncatedSVD(n_components=2).fit_transform(X)
        plot_embedding(X_pca,
                   "Principal Components projection of the digits (time %.2fs)" %
                   (time() - t0))

        #----------------------------------------------------------------------
        # Projection on to the first 2 linear discriminant components

        print("Computing Linear Discriminant Analysis projection")
        X2 = X.copy()
        X2.flat[::X.shape[1] + 1] += 0.01  # Make X invertible
        t0 = time()
        X_lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2).fit_transform(X2, y)
        plot_embedding(X_lda,
                   "Linear Discriminant projection of the digits (time %.2fs)" %
                   (time() - t0))


        #----------------------------------------------------------------------
        # Isomap projection of the digits dataset
        print("Computing Isomap embedding")
        t0 = time()
        X_iso = Isomap(n_neighbors, n_components=2).fit_transform(X)
        print("Done.")
        plot_embedding(X_iso,
                   "Isomap projection of the digits (time %.2fs)" %
                   (time() - t0))


        #----------------------------------------------------------------------
        # Locally linear embedding of the digits dataset
        print("Computing LLE embedding")
        clf = LocallyLinearEmbedding(n_neighbors, n_components=2,
                              method='standard')
        t0 = time()
        X_lle = clf.fit_transform(X)
        print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
        plot_embedding(X_lle,
                   "Locally Linear Embedding of the digits (time %.2fs)" %
                   (time() - t0))


        #----------------------------------------------------------------------
        # Modified Locally linear embedding of the digits dataset
        print("Computing modified LLE embedding")
        clf = LocallyLinearEmbedding(n_neighbors, n_components=2,
                              method='modified')
        t0 = time()
        X_mlle = clf.fit_transform(X)
        print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
        plot_embedding(X_mlle,
                   "Modified Locally Linear Embedding of the digits (time %.2fs)" %
                   (time() - t0))


        #----------------------------------------------------------------------
        # HLLE embedding of the digits dataset
        print("Computing Hessian LLE embedding")
        clf = LocallyLinearEmbedding(n_neighbors, n_components=2,
                              method='hessian')
        t0 = time()
        X_hlle = clf.fit_transform(X)
        print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
        plot_embedding(X_hlle,
                   "Hessian Locally Linear Embedding of the digits (time %.2fs)" %
                   (time() - t0))


        #----------------------------------------------------------------------
        # LTSA embedding of the digits dataset
        print("Computing LTSA embedding")
        clf = LocallyLinearEmbedding(n_neighbors, n_components=2,
                              method='ltsa')
        t0 = time()
        X_ltsa = clf.fit_transform(X)
        print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
        plot_embedding(X_ltsa,
                   "Local Tangent Space Alignment of the digits (time %.2fs)" %
                   (time() - t0))

        #----------------------------------------------------------------------
        # MDS  embedding of the digits dataset
        print("Computing MDS embedding")
        clf = MDS(n_components=2, n_init=1, max_iter=100)
        t0 = time()
        X_mds = clf.fit_transform(X)
        print("Done. Stress: %f" % clf.stress_)
        plot_embedding(X_mds,
                   "MDS embedding of the digits (time %.2fs)" %
                   (time() - t0))

        #----------------------------------------------------------------------
        # Random Trees embedding of the digits dataset
        print("Computing Totally Random Trees embedding")
        hasher = RandomTreesEmbedding(n_estimators=200, random_state=0,
                               max_depth=5)
        t0 = time()
        X_transformed = hasher.fit_transform(X)
        pca = TruncatedSVD(n_components=2)
        X_reduced = pca.fit_transform(X_transformed)

        plot_embedding(X_reduced,
                   "Random forest embedding of the digits (time %.2fs)" %
                   (time() - t0))

        #----------------------------------------------------------------------
        # Spectral embedding of the digits dataset
        print("Computing Spectral embedding")
        embedder = SpectralEmbedding(n_components=2, random_state=0,
                              eigen_solver="arpack")
        t0 = time()
        X_se = embedder.fit_transform(X)

        plot_embedding(X_se,
                   "Spectral embedding of the digits (time %.2fs)" %
                   (time() - t0))

        #----------------------------------------------------------------------
        # t-SNE embedding of the digits dataset
        print("Computing t-SNE embedding")
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        t0 = time()
        X_tsne = tsne.fit_transform(X)

        plot_embedding(X_tsne,
                   "t-SNE embedding of the digits (time %.2fs)" %
                   (time() - t0))

        plt.show()





    # Model Selection Plot
    ###############################
    # overfit-underfit plot for single estimator
    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#sphx-glr-auto-examples-model-selection-plot-validation-curve-py
    def cv_score_compare_plot1d(self, train, target, param_name, param_range, model):
        """ Cross validation score comparision plot
        Single/Multi Features
        Single/Multi Target/Label

        Paramter
        --------

        Built-in Parameter
        ------------------
        score function : accuracy

        Return
        ------
        score vs. a single parameter for a model

        """
        train_scores, test_scores = validation_curve(
            model, train, target, param_name=param_name, param_range=param_range,
            cv=10, scoring="accuracy", n_jobs=1)

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        fig, ax = plt.subplots()

        plt.title("Validation Curve with "+param_name)
        #plt.xlabel("$\gamma$")
        plt.xlabel(str(param_name))
        plt.ylabel("Score")
        plt.ylim(0.0, 1.1)
        lw = 2

        plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
        plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
        plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
        plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
        plt.legend(loc="best")
        if self.native_flag == True:
            plt.show()
        else:
            return ['', mpld3.fig_to_html(fig)]



    #http://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html#sphx-glr-auto-examples-model-selection-plot-underfitting-overfitting-py
    def cv_target_estimated_plot1d(self, train, target, param_model, param_name, param_range, pipeline):
        """ CV estimated single target label plot
        Single/Multi Features
        Single target

        Built-in Parameter
        -----------------
        score function : neg_mean_squared_error

        Return
        ------
        estimated target vs. a single feature
        """
        plt.figure(figsize=(14, 5))
        ax = plt.subplot(1, len(param_range), 1)
        plt.setp(ax, xticks=(), yticks=())

        grid = GridSearchCV(pipeline, param_grid = dict(param_model+'__'+param_name), cv=5 , scoring='precision_macro')
        grid.fit(np.expand_ndims(train, axis=-1), target)

        scores = cross_val_score(pipeline, train, target, scoring="neg_mean_squared_error", cv=10)

        plt.plot(train, grid.predict(np.expand_ndims(train, axis=-1)), label="Model")
        plt.scatter(train, target, label="Samples")
        plt.xlabel(param_name)
        plt.ylabel("y")
        plt.xlim((np.min(param_range), np.max(param_range)))
        plt.ylim((np.min(target), np.max(target)))
        plt.legend(loc="best")
        plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(degrees[i], -scores.mean(), scores.std()))
        plt.show()


