import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import optuna

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


train_path = "C:/Users/82108/Downloads/train.csv/train.csv"
test_path  = "C:/Users/82108/Downloads/test.csv/test.csv"
sub_path   = "C:/Users/82108/Downloads/sample_submission.csv/sample_submission.csv"

train = pd.read_csv(train_path).drop('Id', axis=1)
test  = pd.read_csv(test_path).drop('Id', axis=1)


# 전처리 과정--------------------------------------------------------------------------------

def apply_advanced_preprocessing(df):

    # [기본 결측치 처리]
    df['MonthlyIncome']      = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(0)

    # [이상치 처리] 96, 98 등 비정상 연체 횟수 조정
    bad_cols = [
        'NumberOfTime30-59DaysPastDueNotWorse',
        'NumberOfTimes90DaysLate',
        'NumberOfTime60-89DaysPastDueNotWorse'
    ]
    for col in bad_cols:
        df.loc[df[col] >= 96, col] = df[col].median()

    
    df.loc[df['age'] < 21, 'age'] = df['age'].median()



    df['Total_PastDue'] = (
        df['NumberOfTime30-59DaysPastDueNotWorse'] +
        df['NumberOfTime60-89DaysPastDueNotWorse'] +
        df['NumberOfTimes90DaysLate']
    )
    df['MonthlyDebt']     = df['DebtRatio'] * df['MonthlyIncome']
    df['IncomePerPerson'] = df['MonthlyIncome'] / (df['NumberOfDependents'] + 1)

   
   
    df['Log_Income']      = np.log1p(df['MonthlyIncome'])
    df['Log_Debt']        = np.log1p(df['MonthlyDebt'])
    df['Log_IncomePerP']  = np.log1p(df['IncomePerPerson'])

    
    df['Utilization_Clipped'] = df['RevolvingUtilizationOfUnsecuredLines'].clip(0, 1)

    
    df['Age_Income']      = df['age'] * df['Log_Income']


    df['PastDue_Rate']    = df['Total_PastDue'] / (df['age'] / 12 + 1)

    
    df['Net_Monthly']     = df['MonthlyIncome'] - df['MonthlyDebt']
    df['Log_NetMonthly']  = np.log1p(df['Net_Monthly'].clip(lower=0))

    df['Debt_x_Util']     = df['DebtRatio'] * df['Utilization_Clipped']

    df['Severe_PastDue_Ratio'] = df['NumberOfTimes90DaysLate'] / (df['Total_PastDue'] + 1)

    return df


train_df = apply_advanced_preprocessing(train.copy())
test_df  = apply_advanced_preprocessing(test.copy())

X = train_df.drop('SeriousDlqin2yrs', axis=1)
y = train_df['SeriousDlqin2yrs']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"훈련 샘플: {len(X_train):,} | 검증 샘플: {len(X_val):,}")
print(f"피처 수: {X.shape[1]} (원본 대비 +{X.shape[1] - 11}개 추가)\n")


print("--- [1/3] Optuna LGBM 하이퍼파라미터 탐색 (100 trials) ---")

def lgbm_objective(trial):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 300, 1500),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves':        trial.suggest_int('num_leaves', 31, 255),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'is_unbalance': True,
        'random_state': 42,
        'verbose': -1
    }
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

lgbm_study = optuna.create_study(direction='maximize')
lgbm_study.optimize(lgbm_objective, n_trials=100, show_progress_bar=True)

best_lgbm_params = lgbm_study.best_params
best_lgbm_params.update({'is_unbalance': True, 'random_state': 42, 'verbose': -1})
print(f"LGBM 최적 파라미터: {best_lgbm_params}")
print(f"LGBM 튜닝 후 AUC: {lgbm_study.best_value:.4f}\n")


print("--- [2/3] Optuna XGB 하이퍼파라미터 탐색 (50 trials) ---")

def xgb_objective(trial):
    params = {
        'n_estimators':    trial.suggest_int('n_estimators', 300, 1200),
        'learning_rate':   trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth':       trial.suggest_int('max_depth', 3, 9),
        'subsample':       trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha':       trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda':      trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'scale_pos_weight': 14,
        'eval_metric': 'auc',
        'random_state': 42
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

xgb_study = optuna.create_study(direction='maximize')
xgb_study.optimize(xgb_objective, n_trials=50, show_progress_bar=True)

best_xgb_params = xgb_study.best_params
best_xgb_params.update({'scale_pos_weight': 14, 'eval_metric': 'auc', 'random_state': 42})
print(f"XGB 최적 파라미터: {best_xgb_params}")
print(f"XGB 튜닝 후 AUC: {xgb_study.best_value:.4f}\n")

#최종 모델 확정 -----------------------------------------------------------
models = {
    "LGBM": LGBMClassifier(**best_lgbm_params),
    "XGB":  XGBClassifier(**best_xgb_params),
    "RF":   RandomForestClassifier(
                n_estimators=300,
                class_weight='balanced',
                max_features='sqrt',
                random_state=42
            ),
}


print("--- [3/3] 5-Fold OOF 스태킹 앙상블 학습 ---")

def get_oof_predictions(models_dict, X_full, y_full, X_test_data, n_splits=5):
    """
    데이터 누수 없이 OOF 예측 생성.
    train → 메타 피처로, test → 각 fold 예측값 평균.
    """
    oof_train = np.zeros((len(X_full), len(models_dict)))
    oof_test  = np.zeros((len(X_test_data), len(models_dict)))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for i, (name, model) in enumerate(models_dict.items()):
        test_fold_preds = np.zeros((len(X_test_data), n_splits))

        for j, (tr_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
            X_tr, X_vl = X_full.iloc[tr_idx], X_full.iloc[val_idx]
            y_tr        = y_full.iloc[tr_idx]

            model.fit(X_tr, y_tr)
            oof_train[val_idx, i]  = model.predict_proba(X_vl)[:, 1]
            test_fold_preds[:, j]  = model.predict_proba(X_test_data)[:, 1]

        oof_test[:, i] = test_fold_preds.mean(axis=1)
        fold_auc = roc_auc_score(y_full, oof_train[:, i])
        print(f"  {name} OOF AUC: {fold_auc:.4f}")

    return oof_train, oof_test


oof_train_preds, oof_test_preds = get_oof_predictions(models, X, y, test_df)

#모델 학습 -----------------------------------------------------------------------------------
meta_model = LogisticRegression(C=1.0, random_state=42)
meta_model.fit(oof_train_preds, y)

stacking_auc = roc_auc_score(y, meta_model.predict_proba(oof_train_preds)[:, 1])
print(f"\n스태킹 앙상블 OOF AUC: {stacking_auc:.4f}")

#파일 저장 -------------------------------------------------------------------------------
final_pred = meta_model.predict_proba(oof_test_preds)[:, 1]

submission = pd.read_csv(sub_path)
submission['SeriousDlqin2yrs'] = final_pred
submission.to_csv('final_stacking_submission.csv', index=False)
print("\n[성공] 'final_stacking_submission.csv' 저장 완료")

# 시각화 ----------------------------------------------------------------------------
model_names = list(models.keys()) + ['Stacking Ensemble']
model_aucs  = [
    roc_auc_score(y_val, m.fit(X_train, y_train).predict_proba(X_val)[:, 1])
    for m in models.values()
] + [stacking_auc]

plt.figure(figsize=(10, 6))
bars = sns.barplot(x=model_names, y=model_aucs, palette='muted')
for bar, auc in zip(bars.patches, model_aucs):
    bars.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.001,
        f'{auc:.4f}',
        ha='center', va='bottom', fontsize=11
    )
plt.title('Model Performance Comparison (AUC)', fontsize=14)
plt.ylabel('ROC-AUC')
plt.ylim(min(model_aucs) - 0.01, max(model_aucs) + 0.015)
plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=150)
plt.show()
print("시각화 저장 완료: performance_comparison.png")

# 피처 중요도 --------------------------------------------------------------------
lgbm_final = LGBMClassifier(**best_lgbm_params)
lgbm_final.fit(X, y)

feat_imp = pd.DataFrame({
    'feature':   X.columns,
    'importance': lgbm_final.feature_importances_
}).sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(10, 7))
sns.barplot(data=feat_imp, x='importance', y='feature', palette='viridis')
plt.title('Top 15 Feature Importances (LGBM)', fontsize=14)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()
print("피처 중요도 저장 완료: feature_importance.png")