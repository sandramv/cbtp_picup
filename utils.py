"""Helper functions and constants."""

import pandas as pd
import numpy as np
from tableone import TableOne

def main_cleaning(df):
    # Recode missing data
    df = df.replace(["88", "99", "-88", "-99", "-999", "-888", "888", "999", " ",
                     88, 99, -88, -99, -999, -888, 888, 999], np.nan)

    # Select consent=yes 
    df['consent'] = df['consent'].fillna(1) # assumed the only Ps w/o data consented
    df = df[df['consent'] == 1]

    # Select therapy=cbt
    df['intervention_offered'] = df['intervention_offered'].fillna(1) # assumed nan == CBT
    df = df[df['intervention_offered'] == 1]
    
    return df


def response_to_treatment(outcome_pre, outcome_post, threshold):
    """ Create new binary variable for outcome variable. 
    outcome_pre (int): CHOICE pre-treatment 
    outcome_post (int): post-therapy/follow-up"""
    if ((outcome_post - outcome_pre) / outcome_pre) > threshold:
        return 1
    else:
        return 0

    
def replace_timepoints(data, timepoint_main, timepoint_replace):
    """ Replace with closest timepoint, when main timepoint not available.
    timepoint_main (str): name of timepoint to be filled (e.g. '0')
    timepoint_replace (str): name of timepoint with data (e.g. '2')
    """
    for col in data.loc[:, data.columns.str.endswith(timepoint_main)]:
        new_col = col [:-1] + timepoint_replace
        data[col] = data[col].fillna(df[new_col])    
    return data


def best_components(model, i_repetition, i_fold, df, output_dir):
    """ Extract the fitted PCA components.
    model: fitted PCA model
    i_repetition (int): number of repetition
    i_fold (int): number of cross-validation fold
    df (DataFrame): dataframe
    output_dir (Path): path to save the output
    """
    components = model.components_
    column_names = df.columns
    df_components = pd.DataFrame(components, columns=column_names)
    df_components['iteration'] = str(i_repetition)
    df_components['fold'] = str(i_fold)
    iter_fold = str(i_repetition) + '_' + str(i_fold)
    df_components['iter_fold'] = iter_fold
    df_components['iter_fold'] = iter_fold
    
    explanied_var = model.explained_variance_ratio_
    df_components['explanied_var'] = explanied_var
    
    df_filename = '{:02d}_{:02d}_components.csv'.format(i_repetition, i_fold)
    output_dir = Path(output_dir, 'components')
    output_dir.mkdir(exist_ok=True)
    df_components.to_csv(output_dir/df_filename)


# SELECT RELEVANT VARIABLES
clin_vars = ['Noofmonths1', 'Noofmonths3', 'Noofmonths4', 'Date.2', 'Date.4', 'Primary_diagnosis_classification', 'year_become_unwell']
demog = ['Sex', 'Age', 'BMEvswhite'] 


############## TIMEPOINT 2
# Beliefs & Voice Hearing Rating Scale
psyrats_d_2 = ['pdap.2', 'pddp.2', 'pdc.2', 'pdad.2', 'pdid.2', 'pdd.2']
psyrats_v_2 = ['pvf.2', 'pvd.2', 'pvln.2', 'pvls.2', 'pvb.2', 'pvanc.2', 'pvdnc.2',
             'pvad.2', 'pvid.2', 'pvdn.2', 'pvc.2']
psyrats_2 = psyrats_d_2 + psyrats_v_2


# Warwick-Edinburgh Mental Wellbeing Scale
wemwbs_2 = ['WEMWBS1.2', 'WEMWBS2.2', 'WEMWBS3.2', 'WEMWBS4.2', 'WEMWBS5.2', 'WEMWBS6.2', 'WEMWBS7.2', 'WEMWBS8.2',
          'WEMWBS9.2', 'WEMWBS10.2', 'WEMWBS11.2', 'WEMWBS12.2', 'WEMWBS13.2', 'WEMWBS14.2']


# Work and Social Adjustment Scale
wsas_2 = ['WSAS1.2', 'WSAS2.2', 'WSAS3.2', 'WSAS4.2', 'WSAS5.2']

# Clinical Outcomes in Routine Evaluation
core_2 = ['core.1.2', 'core.2.2', 'core.3.2', 'core.4.2', 'core.5.2', 'core.6.2', 'core.7.2',
        'core.8.2', 'core.9.2', 'core.10.2']

# Depression, Anxiety and Stress
das_2 = ['dass.1.2', 'dass.2.2', 'dass.3.2', 'dass.4.2', 'dass.5.2', 'dass.6.2', 'dass.7.2',
       'dass.8.2', 'dass.9.2', 'dass.10.2', 'dass.11.2', 'dass.12.2', 'dass.13.2', 'dass.14.2',
       'dass.15.2', 'dass.16.2', 'dass.17.2', 'dass.18.2', 'dass.19.2', 'dass.20.2', 'dass.21.2']


# Illness Perception Questionnaire
ipq_2 = ['IPQ1.2', 'IPQ2.2', 'IPQ3.2', 'IPQ4.2', 'IPQ5.2', 'IPQ6.2', 'IPQ7.2',
       'IPQ8.2', 'IPQ9.2', 'IPQ10.2', 'IPQc1.2', 'IPQc2.2', 'IPQc3.2', 'IPQc4.2',
       'IPQc5.2', 'IPQc6.2', 'IPQc7.2', 'IPQc8.2', 'IPQc9.2', 'IPQc10.2', 'IPQc11.2',
       'IPQc12.2']


############## TIMEPOINT 0
# Beliefs & Voice Hearing Rating Scale
psyrats_d_0 = ['pdap.0', 'pddp.0', 'pdc.0', 'pdad.0', 'pdid.0', 'pdd.0']
psyrats_v_0 = ['pvf.0', 'pvd.0', 'pvln.0', 'pvls.0', 'pvb.0', 'pvanc.0', 'pvdnc.0',
             'pvad.0', 'pvid.0', 'pvdn.0', 'pvc.0']
psyrats_0 = psyrats_d_0 + psyrats_v_0

# Warwick-Edinburgh Mental Wellbeing Scale
wemwbs_0 = ['WEMWBS1.0', 'WEMWBS2.0', 'WEMWBS3.0', 'WEMWBS4.0', 'WEMWBS5.0', 'WEMWBS6.0', 'WEMWBS7.0', 'WEMWBS8.0',
          'WEMWBS9.0', 'WEMWBS10.0', 'WEMWBS11.0', 'WEMWBS12.0', 'WEMWBS13.0', 'WEMWBS14.0']
df[wemwbs_0] = df[wemwbs_0].replace([33, 45], [3,4])

# Work and Social Adjustment Scale
wsas_0 = ['WSAS1.0', 'WSAS2.0', 'WSAS3.0', 'WSAS4.0', 'WSAS5.0']

#Clinical Outcomes in Routine Evaluation: Psychological Global Distress (Subjective well-being, Problems/symptoms, Life functioning, Risk/harm
core_0 = ['core.1.0', 'core.2.0', 'core.3.0', 'core.4.0', 'core.5.0', 'core.6.0', 'core.7.0',
        'core.8.0', 'core.9.0', 'core.10.0']

# Depression, Anxiety and Stress
das_0 = ['dass.1.0', 'dass.2.0', 'dass.3.0', 'dass.4.0', 'dass.5.0', 'dass.6.0', 'dass.7.0',
       'dass.8.0', 'dass.9.0', 'dass.10.0', 'dass.11.0', 'dass.12.0', 'dass.13.0', 'dass.14.0',
       'dass.15.0', 'dass.16.0', 'dass.17.0', 'dass.18.0', 'dass.19.0', 'dass.20.0', 'dass.21.0']

# Illness Perception Questionnaire
ipq_0 = ['IPQ1.0', 'IPQ2.0', 'IPQ3.0', 'IPQ4.0', 'IPQ5.0', 'IPQ6.0', 'IPQ7.0',
       'IPQ8.0', 'IPQ9.0', 'IPQ10.0', 'IPQc1.0', 'IPQc2.0', 'IPQc3.0', 'IPQc4.0',
       'IPQc5.0', 'IPQc6.0', 'IPQc7.0', 'IPQc8.0', 'IPQc9.0', 'IPQc10.0', 'IPQc11.0',
       'IPQc12.0']

############## OUTCOME
# Referral
choice_0 = ['CHOICE1A.0', 'CHOICE2A.0', 'CHOICE3A.0', 'CHOICE4A.0', 'CHOICE5A.0', 'CHOICE6A.0', 'CHOICE7A.0',
            'CHOICE8A.0', 'CHOICE9A.0', 'CHOICE10A.0', 'CHOICE11A.0', 'CHOICE12A.0']
df[choice_0] = df[choice_0].replace(40, 4)

# Pre-treatment
choice_2 = ['CHOICE1A.2', 'CHOICE2A.2', 'CHOICE3A.2', 'CHOICE4A.2', 'CHOICE5A.2', 'CHOICE6A.2', 'CHOICE7A.2',
            'CHOICE8A.2', 'CHOICE9A.2', 'CHOICE10A.2', 'CHOICE11A.2', 'CHOICE12A.2']

# Mid-therapy
choice_3 = ['CHOICE1A.3', 'CHOICE2A.3', 'CHOICE3A.3', 'CHOICE4A.3', 'CHOICE5A.3', 'CHOICE6A.3', 'CHOICE7A.3',
            'CHOICE8A.3', 'CHOICE9A.3', 'CHOICE10A.3', 'CHOICE11A.3', 'CHOICE12A.3']

# Post-treatment
choice_4 = ['CHOICE1A.4', 'CHOICE2A.4', 'CHOICE3A.4', 'CHOICE4A.4', 'CHOICE5A.4', 'CHOICE6A.4', 'CHOICE7A.4',
            'CHOICE8A.4', 'CHOICE9A.4', 'CHOICE10A.4', 'CHOICE11A.4', 'CHOICE12A.4'

# Outcome follow-up
choice_5 = ['CHOICE1A.5', 'CHOICE2A.5', 'CHOICE3A.5', 'CHOICE4A.5', 'CHOICE5A.5', 'CHOICE6A.5', 'CHOICE7A.5',
            'CHOICE8A.5', 'CHOICE9A.5', 'CHOICE10A.5', 'CHOICE11A.5', 'CHOICE12A.5']
            
            
# Select data
COLUMN_NAMES = ['Access_number'] + demog + psyrats_0 + wemwbs_0 + wsas_0 + core_0 + das_0 + ipq_0 + psyrats_2 + wemwbs_2 + wsas_2 + core_2 + das_2 + ipq_2 + choice_2 + choice_0 + choice_3 + choice_4 + choice_5 + clin_vars
predictors  = demog + psyrats_2 + core_2 + wemwbs_2 + das_2 + wsas_2 + ipq_2