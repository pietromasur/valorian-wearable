from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import optuna

import seaborn as sns
import scikitplot as skplt
import matplotlib.pyplot as plt

import os
import json


def get_metrics(y, yhat, y_proba, save_path):
    results = dict()
    
    if (y_proba is not None):
        skplt.metrics.plot_roc(y, y_proba)
        plt.savefig(os.path.join(save_path, "roc_auc.png"))
        plt.clf()

    results['macro_f1'] = f1_score(y, yhat, average='macro')
    results['micro_f1'] = f1_score(y, yhat, average='micro')
    results['weighted_f1'] = f1_score(y, yhat, average='weighted')
    results['macro_precision'] = precision_score(y, yhat, average='macro')
    results['micro_precision'] = precision_score(y, yhat, average='micro')
    results['weighted_precision'] = precision_score(y, yhat, average='weighted')
    results['macro_recall'] = recall_score(y, yhat, average='macro')
    results['micro_recall'] = recall_score(y, yhat, average='micro')
    results['weighted_recall'] = recall_score(y, yhat, average='weighted')
    results['macro_roc_auc'] = roc_auc_score(y, y_proba, average='macro', multi_class='ovr')
    results['micro_roc_auc'] = roc_auc_score(y, y_proba, average='micro', multi_class='ovr')
    results['weighted_roc_auc'] = roc_auc_score(y, y_proba, average='weighted', multi_class='ovr')
    results['classification_report'] = classification_report(y, yhat)
    sns.heatmap(confusion_matrix(y, yhat), annot=True, cmap='Blues')
    plt.savefig(os.path.join(save_path, "confusion_matrix.png"))
    plt.clf()
    for key in results:
        print(key, ' ', str(results[key]))
    with open(os.path.join(save_path, "results.json"), "w") as f:
            json.dump(results, f)

    return results


def train(model_name, features_train, target_train, features_val, target_val):
    objective_wrap = lambda trial: _rf_objective(trial, features_train, target_train, features_val, target_val) if model_name=='rf' else _xgb_objective(trial, features_train, target_train, features_val, target_val)
    # Pass func to Optuna studies
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_wrap, n_trials=25)
    return study

def evaluate(study, model_name, save_path, features_train, target_train, features_test, target_test):
    # Get the best hyperparameters
    best_params = study.best_params
    with open(os.path.join(save_path, "params.json"), "w") as f:
        json.dump(best_params, f)
    scaler = RobustScaler()
    
    classificador = RandomForestClassifier(**best_params) if (model_name=='rf') else xgb.XGBClassifier(**best_params)
    # Create the pipeline
    pipeline = make_pipeline(scaler, classificador)
    # Fit the pipeline to training data
    pipeline.fit(features_train, target_train)
    # Make predictions on test data
    all_preds = pipeline.predict(features_test)
    all_preds_proba = pipeline.predict_proba(features_test)
    all_targets = target_test
    # Get the metrics
    get_metrics(all_targets, all_preds, all_preds_proba, save_path)

def train_and_evaluate(model_name, save_path, features_train, target_train, features_val, target_val, features_test, target_test):
    assert model_name in ['rf', 'xgb'],  '{} Modelo não suportado'.format(model_name)
    study = train(model_name, features_train, target_train, features_val, target_val)
    print('F1 no cunjunto de validação para o modelo {}: {}'.format(model_name, study.best_trial.value))
    print("Melhores hiperparâmetros para o modelo {}: {}".format(model_name, study.best_trial.params))
    evaluate(study, model_name, save_path, features_train, target_train,  features_test, target_test)

def _rf_objective(trial, features_train, target_train, features_val, target_val):
    n_estimators = trial.suggest_int('n_estimators', 2, 40)
    max_depth = trial.suggest_int('max_depth', 1, 32)
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    
    scaler = RobustScaler()
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion)
    # Create the pipeline
    pipeline = make_pipeline(scaler, rf)
    # Fit the pipeline to training data
    pipeline.fit(features_train, target_train)
    # Make predictions on val data
    preds = pipeline.predict(features_val)
    metrica = f1_score(target_val, preds, average = 'macro')
    return metrica

def _xgb_objective(trial, features_train, target_train, features_val, target_val):
    xgb_params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300, 50),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
    }
    model = xgb.XGBClassifier(**xgb_params)
    scaler = RobustScaler()
    pipeline = make_pipeline(scaler, model)
    # Fit the pipeline to training data
    pipeline.fit(features_train, target_train)
    # Make predictions on val data
    preds = pipeline.predict(features_val)
    metrica = f1_score(target_val, preds, average = 'macro')
    return metrica