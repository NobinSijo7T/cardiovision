"""
CARDIOVISION - Dataset Tests
"""

import pandas as pd
import pytest

from src.data.label_mapper import assign_labels
from src.data.splitter import patient_level_split

def test_label_mapping():
    # Mock data
    data = {
        'ecg_id': [1, 2, 3],
        'scp_codes': [
            {'NORM': 100.0, 'SR': 0.0},
            {'IMI': 100.0, 'NORM': 10.0},
            {'NDT': 50.0, 'AFIB': 100.0}
        ]
    }
    df = pd.DataFrame(data).set_index('ecg_id')
    
    labeled_df, stats = assign_labels(df)
    
    assert len(labeled_df) == 3
    
    # Priority resolution checks
    assert labeled_df.loc[1, 'label_name'] == 'Normal'
    assert labeled_df.loc[2, 'label_name'] == 'Myocardial Infarction' # MI > Normal
    assert labeled_df.loc[3, 'label_name'] == 'Arrhythmia' # Arrhythmia (AFIB) > STTC (NDT)

def test_patient_split_leakage():
    # Mock data with multiple records per patient
    data = {
        'ecg_id': range(10),
        'patient_id': [1, 1, 2, 3, 3, 4, 5, 5, 6, 7],
        'label': [0, 0, 1, 1, 1, 2, 2, 2, 3, 4],
        'label_name': ['Normal']*2 + ['MI']*3 + ['Arrhythmia']*3 + ['LVH'] + ['STTC']
    }
    df = pd.DataFrame(data).set_index('ecg_id')
    
    df_train, df_val, df_test = patient_level_split(df, 0.6, 0.2, 0.2, random_seed=42)
    
    train_p = set(df_train['patient_id'])
    val_p = set(df_val['patient_id'])
    test_p = set(df_test['patient_id'])
    
    # Check disjoint sets
    assert len(train_p & val_p) == 0
    assert len(train_p & test_p) == 0
    assert len(val_p & test_p) == 0
