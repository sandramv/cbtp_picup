#!/usr/bin/env python3
"""Preprocess data."""


import pandas as pd
import numpy as np
from tableone import TableOne
from pathlib import Path
from utils import main_cleaning, COLUMN_NAMES, replace_timepoints


# Setup directories
PROJECT_ROOT = Path.cwd()
reg_dir = PROJECT_ROOT / 'elasticnet_clf'
reg_dir.mkdir(exist_ok=True)
cv_dir = reg_dir / 'cv'
cv_dir.mkdir(exist_ok=True)

# Read in data
df = pd.read_csv("../data/PICuP_020623.csv", encoding = "ISO-8859-1", low_memory=False)

# Main cleaning (recode missing data, select consent and CBT)
df = main_cleaning(df)

# Select main variables
df = df[COLUMN_NAMES]

# Recode out of range
df['BMEvswhite'] = df['BMEvswhite'].replace(0, 1)
df[psyrats_2] = df[psyrats_2].replace([5, 70], [4, np.nan])
df['WEMWBS13.2'] = df['WEMWBS13.2'].replace([48], 4)
df[das_2] = df[das_2].replace([11,12], 1)
df['IPQc10.2'] = df['IPQc10.2'].replace([55], 5)


# Duration of illness
df['year_become_unwell'] = df['year_become_unwell'].replace(['27'], '2007')
df['year_become_unwell'] = pd.to_datetime(df['year_become_unwell'], format='%Y')
df['Date.2'] = pd.to_datetime(df['Date.2'], dayfirst=True)
df['duration_illness'] = ((df['Date.2'] - df['year_become_unwell']).dt.days)/365

# Duration of therapy
df['Date.2'] = pd.to_datetime(df['Date.2'], dayfirst=True)
df['date.4'] = pd.to_datetime(df['date.4'], dayfirst=True)
df['duration_therapy'] = ((df['date.4'] - df['Date.2']).dt.days)/30

# Primary diagnosis
df['Primary_diagnosis_classification'] = df['Primary_diagnosis_classification'].replace(['5','6','7','9'], 'Other')
df['Primary_diagnosis_classification'] = df['Primary_diagnosis_classification'].replace(['1','2','3','4'], 
                                                                                                  ['Schizophrenia spectrum', 'Bipolar disorder', 
                                                                                                   'Depressive/mood disorder', 'Anxiety disorder'])


# Last observation carried forward
df = replace_timepoints(df, '2', '0')
df = replace_timepoints(df, '4', '3')
df = df.loc[:, ~df.columns.str.endswith('.0')]
df = df.loc[:, ~df.columns.str.endswith('.3')]

# Remove participants with at least one measure (all items) missing
to_dropna = [demog, psyrats_2, wemwbs_2, wsas_2, core_2, das_2, ipq_2, choice_2, choice_4]
for measure in to_dropna:
    df = df.dropna(subset=measure, how='all')

# Replace PSYRATS subscales nan with 0
df[df[psyrats_d_2].isnull().all(axis=1)] = df[df[psyrats_d_2].isnull().all(axis=1)].replace(np.nan, 0)
df[df[psyrats_v_2].isnull().all(axis=1)] = df[df[psyrats_v_2].isnull().all(axis=1)].replace(np.nan, 0)

# Remove features with 0 variance
var_thr = VarianceThreshold(threshold = 0.2)
var_thr.fit(df)
concol = [column for column in df.columns 
          if column not in df.columns[var_thr.get_support()]]

for features in concol:
    print(features)
    
df = df.drop(concol,axis=1)

# Outcome var MODEL POST-THERAPY
df['choice_total_2'] = df[choice_2].mean(axis=1)
df['choice_total_4'] = df[choice_4].mean(axis=1)
df["improvement"] = df.apply(lambda x: response_to_treatment1(x['choice_total_2'], x['choice_total_4'], threshold=.30), axis=1)
df['improvement_post_therapy'] = np.where((df['improvement']==0) & (df['choice_total_4']>=7), 1, df['improvement_post_therapy'])

# Save cleaned df
df.to_csv("../data/PICuP_data_cleaned.csv", index=False)

# Generate Table 1 demographics
columns = ['Sex', 'Age', 'BMEvswhite', 'duration_illness', 'duration_therapy', 'Primary_diagnosis_classification']
categorical = ['Sex', 'BMEvswhite', 'Primary_diagnosis_classification']
groupby = 'improvement_post_therapy'
labels = {"BMEvswhite": "Ethnicity",
          'duration_illness': "Duration of illness",
          'duration_therapy': "Duration of therapy",
          'Primary_diagnosis_classification': "Diagnosis",
          "improvement_post_therapy": "Improvement vs non-improved at post-therapy"}

mytable = TableOne(df, columns, categorical, groupby, rename=labels, pval=True)
print(mytable.tabulate(tablefmt="fancy_grid"))
mytable.to_excel('./Table 1. Demographics_model.xlsx')