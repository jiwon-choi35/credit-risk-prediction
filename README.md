# credit-risk-prediction
# Give Me Some Credit — 신용 불이행 예측 모델

Kaggle **Give Me Some Credit** 대회용 앙상블 머신러닝 파이프라인입니다.  
2년 내 심각한 연체(`SeriousDlqin2yrs`) 발생 확률을 예측하며, ROC-AUC를 기준 지표로 사용합니다.

---

## 결과 요약

| 모델 | AUC |
|------|-----|
| Random Forest (baseline) | 0.8327 |
| XGBoost (baseline) | 0.8388 |
| LightGBM (baseline) | 0.8619 |
| **Stacking Ensemble (v2)** | **0.8637** |

---

## 파일 구조

```
├── givemesomecredit_v2.py     # 메인 실행 파일
├── final_stacking_submission.csv  # 최종 제출 파일 (실행 후 생성)
├── performance_comparison.png     # 모델 성능 비교 차트 (실행 후 생성)
├── feature_importance.png         # 피처 중요도 차트 (실행 후 생성)
└── README.md
```

---

## 요구사항

### Python 패키지 설치

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm optuna
```

### 데이터 준비

[Kaggle 대회 페이지](https://www.kaggle.com/c/GiveMeSomeCredit)에서 아래 파일을 다운로드합니다.

```
train.csv
test.csv
sample_submission.csv
```



```python
train_path = r"C:\Users\...\train.csv"
test_path  = r"C:\Users\...\test.csv"
sub_path   = r"C:\Users\...\sample_submission.csv"
```

---

## 실행 방법

```bash
python givemesomecredit_v2.py
```



```python
lgbm_study.optimize(lgbm_objective, n_trials=20)   
xgb_study.optimize(xgb_objective,  n_trials=10)    
```

---

## 파이프라인 구조

### 1. 전처리

- `MonthlyIncome` 결측치 → 중앙값 대체
- `NumberOfDependents` 결측치 → 0으로 대체
- 연체 횟수 96/98 이상 → 이상치로 간주하여 중앙값 대체
- 21세 미만 나이 → 이상치로 간주하여 중앙값 대체

### 2. 피처 엔지니어링

기존 3개 파생 변수에서 **11개 추가** (총 21개 피처로 확장)

| 피처 | 설명 |
|------|------|
| `Total_PastDue` | 전체 연체 횟수 합계 |
| `MonthlyDebt` | 월 부채 금액 (DebtRatio × 소득) |
| `IncomePerPerson` | 인당 월 소득 |
| `Log_Income` | 월 소득 로그 변환 |
| `Log_Debt` | 월 부채 로그 변환 |
| `Utilization_Clipped` | 신용 활용도 (0~1 클리핑) |
| `Age_Income` | 나이 × 로그 소득 상호작용 |
| `PastDue_Rate` | 연 평균 연체율 |
| `Net_Monthly` | 실질 가처분소득 |
| `Debt_x_Util` | 부채비율 × 신용활용도 복합 지표 |
| `Severe_PastDue_Ratio` | 90일+ 심각 연체 비중 |

### 3. 하이퍼파라미터 최적화 (Optuna)

- **LightGBM**: 100 trials, TPE 샘플러
- **XGBoost**: 50 trials, TPE 샘플러
- 탐색 범위: learning rate, num_leaves, subsample, reg_alpha/lambda 등

### 4. 스태킹 앙상블

단순 가중 평균 대신 OOF(Out-of-Fold) 스태킹을 적용합니다.

```
[LightGBM]  ─┐
[XGBoost]   ─┼─▶ OOF 예측 (5-Fold) ─▶ Logistic Regression (메타 모델) ─▶ 최종 예측
[RandomForest]┘
```

- 5-Fold Stratified K-Fold로 데이터 누수 방지
- 메타 모델로 Logistic Regression 사용

---

## 출력물

실행 완료 후 아래 파일이 생성됩니다.

- `final_stacking_submission.csv` 
- `performance_comparison.png` — 모델별 AUC 비교 막대 차트
- `feature_importance.png` — LGBM 기준 상위 15개 피처 중요도
