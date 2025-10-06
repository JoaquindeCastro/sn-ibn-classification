# IMPORTS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,ConfusionMatrixDisplay,classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz
from astropy.cosmology import Planck18 as cosmo


# SETTINGS

ROOT_DIR = r'C:\Users\jgmad\Research\Ibn'
DATA_DIR =  os.path.join(ROOT_DIR, "data")
summary_file, = glob(os.path.join(DATA_DIR, "ZTFBTS_summary.csv"))
summary_data = pd.read_csv(summary_file)
summary_data.replace('-', np.nan, inplace=True)
param_file, = glob(os.path.join(DATA_DIR, f"SN_interpretable_params{'_'+str(DAYS_AFTER) if DAYS_AFTER else ''}.csv"))
df = pd.read_csv(param_file)

# CONTROLS

DAYS_AFTER = 10

# FILTERING


try:
    df = df.rename(columns={'oid':'supernova_name'})
    df = df.drop(['oid_r','oid_g'],axis=1)
except KeyError:
    pass

df['first_det'] = df[['first_det_r','first_det_g']].min(axis=1)

lookup_dict = dict(zip(summary_data['ZTFID'], summary_data['type'] == 'SN Ibn'))
z_dict = dict(zip(summary_data['ZTFID'], summary_data['redshift']))
type_dict = dict(zip(summary_data['ZTFID'], summary_data['type']))

df['Ibn'] = df['supernova_name'].map(lookup_dict)
df['redshift'] = df['supernova_name'].map(z_dict).astype(float)

df = df[~df['supernova_name'].map(type_dict).isnull()] # For now only labeled dataset
print(f'Before redshift cut: {len(df)}')
df = df[~df['redshift'].isnull()]
df = df[df['redshift'] > 0]
print(f'After redshift cut {len(df)}')

slope_features = ['rise_slope_r','rise_slope_g','decline_slope_r','decline_slope_g']

dropped_features = ['meandec_g','meandec_r','meanra_g','meanra_r','last_nondet_g','last_nondet_r','s0_g','s0_r','rise_time_flag_r','rise_time_flag_g']
imputed_features = slope_features+['duration_g','duration_r','peak_epoch_g', 'peak_epoch_r','rise_time_g', 'rise_time_r']
safe_features = [ 'first_det', 'first_det_g', 'first_det_r',
       'ndetection_g', 'ndetection_r',
       'color'] #+  [n+'_missing' for n in imputed_features]
cut_features = ['peak_mag_r', 'peak_mag_g','redshift']

features = imputed_features + cut_features + safe_features

# Drop columns in droppped_features
df = df.drop(dropped_features,axis=1)

# Drop NaN values in cut features
df.replace(-9999, np.nan, inplace=True)
print(f'Before cut_features cut {len(df)}')
df = df.dropna(subset=cut_features)
print(f'After cut_features cut {len(df)}')

# Impute features in imputed_features
from sklearn.impute import SimpleImputer
for col in imputed_features:
    df[f'{col}_missing'] = df[col].isna().astype(int)
imputer = SimpleImputer(strategy='median')
df[imputed_features] = imputer.fit_transform(df[imputed_features])

# Add color g - r

df['color'] = df['peak_mag_g'] - df['peak_mag_r']

# Correct to peak mag from redshift:

dL = cosmo.luminosity_distance(df['redshift']).to("pc").value
mu = 5 * np.log10(dL) - 5           
df['peak_mag_r'] = df['peak_mag_r'] - mu
df['peak_mag_g'] = df['peak_mag_g'] - mu

df['peak_epoch_r'] = df['peak_epoch_r'] - df['first_det_r']
df['peak_epoch_g'] = df['peak_epoch_g'] - df['first_det_g']
print(len(df))

# UTILITIES

prefixes = ['SN','AT', 'TD']
def strip_name(name):
    if name == '-':
        return None
    elif "".join(name[:2]) in prefixes:
        return name[2:]
    else:
        return name
    
# SPLITTING

df_ibn = df[df['supernova_name'].isin(summary_data[summary_data['type'] == 'SN Ibn']['ZTFID'])].copy()

# split by Ibn

df_rest = df[df['Ibn'] == False]
print('Before',len(df))
print('After',len(df_rest))
print('Ibn',len(df_ibn))

# LABELING

df_rest_filtered = df_rest.copy()
df_ibn_filtered = df_ibn.copy()

# FULL DATASET

df_rest_filtered['Ibn'] = 0
df_ibn_filtered['Ibn'] = 1

full = pd.concat([df_rest_filtered,df_ibn_filtered],ignore_index=True)

# TRAIN TEST SPLIT

unique_SN = full['supernova_name'].unique()


SN_to_type = dict(zip(full['supernova_name'], full['Ibn']))
types = [SN_to_type[SN] for SN in unique_SN]

print(unique_SN,SN_to_type,types)

train_SN,test_SN = train_test_split(unique_SN,stratify=types,test_size=0.4,random_state=12282005)

train_mask = full['supernova_name'].isin(train_SN)
test_mask = ~train_mask

X_train = full[train_mask][features]
y_train = full[train_mask]['Ibn']

X_test = full[test_mask][features]
y_test = full[test_mask]['Ibn']
print(len(full),len(X_test))

# TUNING

# OPTIMIZING FOR F BETA SCORE 

# Try to tune parameters using optuna

import optuna
from lightgbm import LGBMClassifier,early_stopping, log_evaluation
from sklearn.metrics import fbeta_score #f1_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

import warnings
warnings.filterwarnings("ignore")

optuna.logging.set_verbosity(optuna.logging.FATAL)

from imblearn.over_sampling import SMOTE

alpha = 1/2 # 1/beta

full = full.drop_duplicates(subset=['supernova_name']).reset_index(drop=True)

X = full[features]
y = full['Ibn']

# Define objective function for Optuna
def objective_fn(trial):
    sampling_strategy = trial.suggest_uniform('sampling_strategy', 0.3, 1.0)
    threshold = trial.suggest_uniform('threshold', 0.05, 0.95)

    params = {
        'class_weight': 'balanced',
        'n_estimators': 1000,
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'random_state': 12282005,
        'verbosity':-1,
        #'objective':focal_obj
    }

    # want to do training and validation over 5 different splits:
    sgkf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=12282005
    )
    mscores = []
    scores = []

    sm = SMOTE(random_state=12282005,sampling_strategy=sampling_strategy)

    # We have to apply SMOTE within each fold so we start with X and y

    for train_idx,val_idx in sgkf.split(X,y):
        X_train_fold,y_train_fold,X_val_fold,y_val_fold = X.iloc[train_idx],y.iloc[train_idx],X.iloc[val_idx],y.iloc[val_idx]

        X_train_fold_resampled,y_train_fold_resampled = sm.fit_resample(X_train_fold,y_train_fold)
        #X_train_fold_resampled,y_train_fold_resampled = X_train_fold,y_train_fold

        model = LGBMClassifier(**params)

        model.fit(
            X_train_fold_resampled, y_train_fold_resampled,
            eval_set=[(X_val_fold, y_val_fold)],
            #eval_metric=focal_eval,
            callbacks=[early_stopping(100), log_evaluation(0)]
        )

        y_probs = model.predict_proba(X_val_fold)[:, 1]
        y_pred = (y_probs >= threshold).astype(int) 
        #score = f1_score(y_test, y_pred, average='binary')  # Focus on class 1
        if alpha:
            score = fbeta_score(y_val_fold, y_pred, beta=1/alpha, pos_label=1, average='binary')  # Focus on class 1 (Ibn)
            mscore = recall_score(y_val_fold, y_pred,pos_label=1,average='binary')
        else:
            mscore = fbeta_score(y_val_fold, y_pred, beta=2, pos_label=1, average='binary')  # Focus on class 1 (Ibn)
            score = recall_score(y_val_fold, y_pred,pos_label=1,average='binary')
        mscores.append(mscore)
        scores.append(score)

    scores_mean = np.mean(scores)
    scroes_std  = np.std(scores, ddof=1)
    trial.set_user_attr('sampling_strategy', sampling_strategy)
    trial.set_user_attr('threshold', threshold)
    #trial.set_user_attr('alpha', alpha)
    #trial.set_user_attr('gamma', gamma)
    trial.set_user_attr('scores_std', float(scroes_std))
    return scores_mean

# Run Optuna optimization
study = optuna.create_study(direction='maximize')

initial_params = {
    'sampling_strategy': 0.9332037435470149,
    'threshold': 0.06277550248785539,
    'learning_rate': 0.06940600855324495,
    'num_leaves': 26,
    'max_depth': 4,
    'min_child_samples': 84,
    'reg_alpha': 0.2662044019333212,
    'reg_lambda': 0.09337327992988727,
    'subsample': 0.6666028791233275,
    'colsample_bytree': 0.6652640151318506

}

study.enqueue_trial(initial_params)
study.optimize(objective_fn, n_trials=500)

# Best params and model
print("Best trial:")
print(f"  Value: {study.best_trial.value}")
print(f"Standard Deviation: {study.best_trial.user_attrs['scores_std']}")
print(f"  Params: ")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")

# ---

# FINAL TRAINING

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Get best params from study
best_params = study.best_trial.params

# Add necessary fixed parameters
best_params.update({
    'class_weight': 'balanced',
    'n_estimators': 1000,
    'random_state': 12282005,
    #'objective':focal_loss_lgb(study.best_trial.user_attrs['alpha'],study.best_trial.user_attrs['gamma'])
})

# Initialize model
lgbm = LGBMClassifier(**best_params)

sm = SMOTE(random_state=12282005,sampling_strategy=study.best_trial.user_attrs['sampling_strategy'])

X_train_resampled,y_train_resampled = sm.fit_resample(X_train,y_train)
#X_train_resampled,y_train_resampled = X_train,y_train

# Train model with early stopping
lgbm.fit(
    X_train_resampled, y_train_resampled,
    eval_set=[(X_test, y_test)],
    #eval_metric=focal_loss_eval(study.best_trial.user_attrs['alpha'],study.best_trial.user_attrs['gamma']),
    callbacks=[early_stopping(1000), log_evaluation(0)]
)

# Predict
y_probs = lgbm.predict_proba(X_test)[:, 1]
y_pred = (y_probs >= study.best_trial.user_attrs['threshold']).astype(int) 

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

results_csv_file = os.path.join(DATA_DIR, "interpretable_results.csv")
header_fields = ['MODEL','METRIC','SC0RE','STD','DAYS_AFTER']
file_exists = os.path.exists(results_csv_file)

# FINAL CROSS VALIDATION

from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.pipeline import make_pipeline
from sklearn.metrics import recall_score, precision_score
import numpy as np

try:
    best_threshold = best_params.pop('threshold')  # Extract threshold separately
    print(best_threshold)
except KeyError:
    pass

try:
    sampling_strategy = best_params.pop('sampling_strategy')
    print(sampling_strategy)
except KeyError:
    pass

# 1) Reconstruct your best‐param model
best_params.update({
    'class_weight': 'balanced',
    'n_estimators':   1000,
    'random_state':   12282005,
})

clf = LGBMClassifier(**best_params)
sm = SMOTE(random_state=12282005,sampling_strategy=sampling_strategy)

# 3) Define the same CV splitter you used in Optuna
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=12282005)

recalls = []
precisions = []
fprs = []
conf_matrices = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

    model = LGBMClassifier(**best_params)
    model.fit(X_train_resampled, y_train_resampled)

    y_probs = model.predict_proba(X_val)[:, 1]
    y_pred_fold = (y_probs >= best_threshold).astype(int)

    recall = recall_score(y_val, y_pred_fold)
    precision = precision_score(y_val,y_pred_fold)
    cm = confusion_matrix(y_val, y_pred_fold)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred_fold).ravel()

    recalls.append(recall)
    precisions.append(precision)
    conf_matrices.append(cm)
    fprs.append(fp/(tn+fp))

    print(f"\nFold {fold+1} Confusion Matrix:\n{cm}")
    print(f"Recall: {recall:.4f}")

print("\nRecall per fold:", recalls)
print("\nPrecisions per fold:", precisions)
print("Mean recall:   ", np.mean(recalls))
print("Mean precision:   ", np.mean(precisions))
print("STD of recall: ", np.std(recalls, ddof=1))
print("STD of precision: ", np.std(precisions, ddof=1))
print("Mean FPR:   ", np.mean(fprs))
print("STD of recall: ", np.std(fprs, ddof=1))