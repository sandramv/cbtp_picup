#!/usr/bin/env python3
"""Post therapy models: run Post-therapy models on follow-data."""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix



# Load saved models
PROJECT_ROOT = Path.cwd()
components_dir = PROJECT_ROOT / 'elasticnet_clf/cv'

# Load saved models
pcas = []
for path in sorted(components_dir.glob('*pca.joblib')):
    f = joblib.load(path)
    pcas.append(f)

models = []
for path in sorted(components_dir.glob('*regressor.joblib')):
    f = joblib.load(path)
    models.append(f)


# Read in data
df = pd.read_csv("../data/PICuP_data_cleaned.csv")

# Select participants with follow-up outcome (CHOICE 5)
df = df.dropna()
df = df.dropna(subset=choice_5, how='all')

# Outcome var
df['choice_total_2'] = df[choice_2].mean(axis=1)
df['choice_total_5'] = df[choice_5].mean(axis=1)
df["improvement_fu"] = df.apply(lambda x: response_to_treatment1(x['choice_total_2'], x['choice_total_5'], threshold=.30), axis=1)
df['improvement_fu'] = np.where((df['improvement_fu']==0) & (df['choice_total_5']>=7), 1, df['improvement_fu'])

# Ensemble soft voting
X = df.iloc[:, 1:-38].values
y = df['improvement_fu'].values

predictions = []
n_folds=100

for i in range(n_folds):
   
    p = pcas[i]
    m = models[i]
    
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X)
    X_test = p.transform(X_test)
    test_predictions = m.predict_proba(X_test)
    predictions.append(test_predictions[:,1])
    
predictions = np.array(predictions)
predictions_mean = np.mean(predictions, axis=0)

cm = confusion_matrix(y, predictions_mean>0.5)
bac = np.true_divide(np.sum(np.true_divide(np.diagonal(cm), np.sum(cm, axis=1))), cm.shape[1])
sens = np.true_divide(cm[1, 1], np.sum(cm[1, :]))
spec = np.true_divide(cm[0, 0], np.sum(cm[0, :]))
ppv = np.true_divide(cm[0, 0], np.sum(cm[:, 0]))
npv = np.true_divide(cm[1, 1], np.sum(cm[:, 1]))

print("Balanced accuracy = ", bac)
print("Sensitivity = ", sens)
print("Specificity = ", spec)
print("PPV = ", ppv)
print("NPV = ", npv)