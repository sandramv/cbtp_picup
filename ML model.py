#!/usr/bin/env python3
"""
Elasticnet model to predict recovery.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from utils import predictors

# Read in data
df = pd.read_csv("../data/PICuP_data_cleaned.csv")

# Predictors and label
X = df[predictors].values
y = df['improvement_post_therapy'].values

# Model paramters
n_repetitions = 20
n_folds = 5
n_nested_folds = 5
knn_n_neighbors = 15
pca_n_components = 30

cv_bac = {}
cv_sens = {}
cv_spec = {}
cv_ppv = {}
cv_npv = {}
cv_coef = {}
    
for i_repetition in range(n_repetitions):

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for i_fold, (train_index, test_index) in enumerate(kf.split(X, y)):

        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Normalize data #
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # Data imputation
        imp = KNNImputer(n_neighbors=knn_n_neighbors, weights="uniform")
        x_train = imp.fit_transform(x_train)
        x_test = imp.transform(x_test)

        # Choose PCA or FA
        transformer = PCA(n_components=pca_n_components)
        x_train = transformer.fit_transform(x_train)
        x_test = transformer.transform(x_test)

        best_components(transformer, i_repetition, i_fold, df[features], './elasticnet_clf/')

        # Grid search
        reg = linear_model.LogisticRegression(solver='saga', penalty='elasticnet', max_iter=1000)

        search_space = {'l1_ratio': np.arange(0, 1, 0.2)}

        nested_kf = KFold(n_splits=n_nested_folds, random_state=seed, shuffle=True)
        gridsearch = GridSearchCV(estimator=reg,
                                  param_grid=search_space,
                                  scoring='balanced_accuracy',
                                  refit=True, cv=nested_kf,
                                  verbose=0, n_jobs=1)
        gridsearch.fit(x_train, y_train)

        best_svm = gridsearch.best_estimator_
        params_results = {'means': gridsearch.cv_results_['mean_test_score'],
                          'params': gridsearch.cv_results_['params']}

        predictions = best_svm.predict(x_test)
        predictions_train = best_svm.predict(x_train)

        cm = confusion_matrix(y_test, predictions)
        tn, fp, fn, tp = cm.ravel()
        bac_test = balanced_accuracy_score(y_test, predictions)
        sens_test = tp / (tp + fn)
        spec_test = tn / (tn + fp)
        ppv_test = tn / (tn + fn)
        npv_test = tp / (tp + fp)

        cv_bac[i_repetition, i_fold] = bac_test
        cv_sens[i_repetition, i_fold] = sens_test
        cv_spec[i_repetition, i_fold] = spec_test
        cv_ppv[i_repetition, i_fold] = ppv_test
        cv_npv[i_repetition, i_fold] = npv_test

        # Save scaler, model,  model parameters and model scores
        scaler_filename = '{:02d}_{:02d}_scaler.joblib'.format(i_repetition, i_fold)
        pca_filename = '{:02d}_{:02d}_pca.joblib'.format(i_repetition, i_fold)
        model_filename = '{:02d}_{:02d}_regressor.joblib'.format(i_repetition, i_fold)
        params_filename = '{:02d}_{:02d}_params.joblib'.format(i_repetition, i_fold)
        scores_array = np.array([bac_test, sens_test, spec_test])
        scores_filename = '{:02d}_{:02d}_scores.npy'.format(i_repetition, i_fold)

        joblib.dump(scores_array,cv_dir / scores_filename)
        joblib.dump(best_svm, cv_dir / model_filename)
        joblib.dump(scaler, cv_dir / scaler_filename)
        joblib.dump(transformer, cv_dir / pca_filename)
        
        # Save coefficients
        cv_coef[i_repetition, i_fold] = np.ravel(best_svm.coef_)

# Variables for CV means across all models
print('CV results')
print('Bac: Mean(SD) = %.3f(%.3f)' % (np.array(list(cv_bac.values())).mean(), np.array(list(cv_bac.values())).std()))
print('Sens: Mean(SD) = %.3f(%.3f)' % (np.array(list(cv_sens.values())).mean(), np.array(list(cv_sens.values())).std()))
print('Spec: Mean(SD) = %.3f(%.3f)' % (np.array(list(cv_spec.values())).mean(), np.array(list(cv_spec.values())).std()))
print('PPV: Mean(SD) = %.3f(%.3f)' % (np.array(list(cv_ppv.values())).mean(), np.array(list(cv_ppv.values())).std()))
print('NPV: Mean(SD) = %.3f(%.3f)' % (np.array(list(cv_npv.values())).mean(), np.array(list(cv_npv.values())).std()))