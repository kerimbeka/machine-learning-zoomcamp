import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, roc_auc_score, recall_score

import xgboost as xgb

from imblearn.combine import SMOTEENN

import optuna

def prepare_data():
    #loading data
    df = pd.read_csv("./data/data.csv")

    #removing space in column naming
    df.columns = df.columns.str.strip()

    #selected columns after eda
    selected_columns = ['Bankrupt?',
                        'Continuous interest rate (after tax)',
                        'Debt ratio %',
                        'Degree of Financial Leverage (DFL)',
                        'Current Liability to Current Assets', 
                        'Equity to Liability',
                        'Interest Coverage Ratio (Interest expense to EBIT)',
                        'Interest Expense Ratio', 
                        "Net Income to Stockholder's Equity",
                        'Net Income to Total Assets',
                        'Persistent EPS in the Last Four Seasons']
    
    df = df[selected_columns]

    # splitting data
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1, stratify=df['Bankrupt?'])
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1, stratify=df_full_train['Bankrupt?'])

    df_full_train = df_full_train.reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_full_train = df_full_train['Bankrupt?'].values
    y_train = df_train['Bankrupt?'].values
    y_val = df_val['Bankrupt?'].values
    y_test = df_test['Bankrupt?'].values

    del df_full_train['Bankrupt?']
    del df_train['Bankrupt?']
    del df_val['Bankrupt?']
    del df_test['Bankrupt?']

    return df_full_train, df_train, df_val, df_test, y_full_train, y_train, y_val, y_test



def train():
    
    df_full_train, df_train, df_val, df_test, y_full_train, y_train, y_val, y_test = prepare_data()

    def objective(trial):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'eta': trial.suggest_float('eta', 0.005, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1),
            'gamma': trial.suggest_float('gamma', 0.001, 1.0, log=True),
            'n_jobs': trial.suggest_int('n_jobs', -1, -1),
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            }

        #applying resampling method
        sm = SMOTEENN(random_state=42)
        df_train_res, y_train_res = sm.fit_resample(df_train, y_train)
    
        #training the model
        model = xgb.XGBClassifier(**params)
        model.fit(df_train_res, y_train_res)
        
        #evaluating recall score on val set
        y_pred = model.predict(df_val)
        score = recall_score(y_val, y_pred)

        return score
    
    # Run the hyperparameter optimization with Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print("Best trial:", study.best_trial.params)
    print("Best score:", round(study.best_value, 3))

    #applying resample method
    sm = SMOTEENN(random_state=42)
    df_full_train_res, y_full_train_res = sm.fit_resample(df_full_train, y_full_train)

    #training the model with the best parameters
    model = xgb.XGBClassifier(**study.best_trial.params)
    model.fit(df_full_train_res, y_full_train_res)

    #saving the model
    model.save_model("./model/model.json")

    #evaluating the model on test set
    y_pred_proba = model.predict_proba(df_test)
    y_pred = model.predict(df_test)
    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])

    print("Classification report:")
    print(classification_report(y_test, y_pred))
    
    print("Roc auc score on test set:", round(roc_auc, 3))

if __name__ == "__main__":
    train()
