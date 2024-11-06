# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error

## 필요한 데이터 불러오기
blueberry_train=pd.read_csv("./data/wildblueberry/train.csv")
blueberry_test=pd.read_csv("./data/wildblueberry/test.csv")
sub_df=pd.read_csv("./data/wildblueberry/sample_submission.csv")

# id: 고유 식별자 (int64)
# clonesize: 클론 크기 (float64)
# honeybee: 꿀벌의 밀도 (float64)
# bumbles: 땅벌의 밀도 (float64)
# andrena: 안드레나 밀도 (float64)
# osmia: 오스미아 밀도 (float64)
# MaxOfUpperTRange: 상위 온도 범위의 최대값 (float64)
# MinOfUpperTRange: 상위 온도 범위의 최소값 (float64)
# AverageOfUpperTRange: 상위 온도 범위의 평균값 (float64)
# MaxOfLowerTRange: 하위 온도 범위의 최대값 (float64)
# MinOfLowerTRange: 하위 온도 범위의 최소값 (float64)
# AverageOfLowerTRange: 하위 온도 범위의 평균값 (float64)
# RainingDays: 강수 일수 (float64)
# AverageRainingDays: 평균 강수 일수 (float64)
# fruitset: 과실 형성율 (float64)
# fruitmass: 과실 질량 (float64)
# seeds: 씨앗 수 (float64)
# yield: 목표 변수로, 야생 블루베리의 생산량 (float64)

# blueberry_train의 정보를 알아보자. # 모두 숫자형, null값 없음, 18개의 열
blueberry_train.info()

# train_X, train_y 구분
blueberry_train_X = blueberry_train.drop(columns=["id", "yield"])
blueberry_train_y = blueberry_train[["yield"]]

blueberry_train_X
blueberry_train_y

# 교차 검증 설정
kf = KFold(n_splits=15, shuffle=True, random_state=2024)

# n_jobs = -1, cpu에 작업을 각각 할당
def rmse(model):
    score = np.sqrt(-cross_val_score(model, blueberry_train_X, blueberry_train_y, cv = kf,
                                     n_jobs = -1, 
                                     scoring = "neg_mean_squared_error").mean())
    return(score)

lasso = Lasso(alpha=0.01)
rmse(lasso)

# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(0, 700, 10)
mean_scores = np.zeros(len(alpha_values))

k=0
for alpha in alpha_values:
    lasso = Lasso(alpha=alpha)
    mean_scores[k] = rmse(lasso)
    k = k + 1

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})
df

# 결과 시각화
plt.plot(df['lambda'], df['validation_error'], label='Validation Error', color='red')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Lasso Regression Train vs Validation Error')
plt.show()

# 최적의 alpha 값 찾기 
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)

# ----------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split

## 필요한 데이터 불러오기
blueberry_train = pd.read_csv("./data/wildblueberry/train.csv")
blueberry_test = pd.read_csv("./data/wildblueberry/test.csv")
sub_df = pd.read_csv("./data/wildblueberry/sample_submission.csv")

# train_X, train_y 구분
blueberry_train_X = blueberry_train.drop(columns=["id", "yield"])
blueberry_train_y = blueberry_train[["yield"]]

# 교차 검증 설정
kf = KFold(n_splits=15, shuffle=True, random_state=2024)

# n_jobs = -1, cpu에 작업을 각각 할당
def rmse(model):
    score = np.sqrt(-cross_val_score(model, blueberry_train_X, blueberry_train_y, cv=kf,
                                     n_jobs=-1, 
                                     scoring="neg_mean_squared_error").mean())
    return score

# Lasso와 Ridge 모델에 대한 검증 오류 계산
alpha_values = np.logspace(-8, 1, 100) 
lasso_scores = np.zeros(len(alpha_values))
ridge_scores = np.zeros(len(alpha_values))

for i, alpha in enumerate(alpha_values):
    lasso = Lasso(alpha=alpha)
    ridge = Ridge(alpha=alpha)
    lasso_scores[i] = rmse(lasso)
    ridge_scores[i] = rmse(ridge)

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'lasso_validation_error': lasso_scores,
    'ridge_validation_error': ridge_scores
})

# 결과 시각화
# plt.plot(df['lambda'], df['lasso_validation_error'], label='Lasso Validation Error', color='red')
# plt.plot(df['lambda'], df['ridge_validation_error'], label='Ridge Validation Error', color='blue')
# plt.xlabel('Lambda')
# plt.ylabel('Mean Squared Error')
# plt.legend()
# plt.title('Lasso vs Ridge Regression Train vs Validation Error')
# plt.show()

# 최적의 alpha 값 찾기
optimal_lasso_alpha = df['lambda'][np.argmin(df['lasso_validation_error'])]
optimal_ridge_alpha = df['lambda'][np.argmin(df['ridge_validation_error'])]

print("Optimal Lasso lambda:", optimal_lasso_alpha)
print("Optimal Ridge lambda:", optimal_ridge_alpha)

# Optimal Lasso lambda: 0.005336699231206312
# Optimal Ridge lambda: 0.023101297000831626

# 최적의 alpha 값을 가진 Lasso와 Ridge 모델 정의
lasso = Lasso(alpha=optimal_lasso_alpha)
ridge = Ridge(alpha=optimal_ridge_alpha)

# OLS 모델 정의
ols = LinearRegression()

# 배깅 모델 정의
bagging_lasso = BaggingRegressor(estimator=lasso, n_estimators=30, random_state=2024)
bagging_ridge = BaggingRegressor(estimator=ridge, n_estimators=30, random_state=2024)
bagging_ols = BaggingRegressor(estimator=ols, n_estimators=30, random_state=2024)

# 모델 훈련
bagging_lasso.fit(blueberry_train_X, blueberry_train_y)
bagging_ridge.fit(blueberry_train_X, blueberry_train_y)
bagging_ols.fit(blueberry_train_X, blueberry_train_y)

# 테스트 데이터 준비
blueberry_test_X = blueberry_test.drop(columns=["id"])

# 각 모델의 예측 수행
lasso_predictions = bagging_lasso.predict(blueberry_test_X)
ridge_predictions = bagging_ridge.predict(blueberry_test_X)
ols_predictions = bagging_ols.predict(blueberry_test_X)

# 앙상블: 세 모델의 예측값을 평균화
ensemble_predictions = (lasso_predictions + ridge_predictions + ols_predictions) / 3

# 앙상블 예측 결과 저장
sub_df["yield"] = ensemble_predictions
sub_df.to_csv("./data/wildblueberry/submission_ensemble_bagging.csv", index=False) 


# ------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split

## 필요한 데이터 불러오기
blueberry_train = pd.read_csv("./data/wildblueberry/train.csv")
blueberry_test = pd.read_csv("./data/wildblueberry/test.csv")
sub_df = pd.read_csv("./data/wildblueberry/sample_submission.csv")

# train_X, train_y 구분
blueberry_train_X = blueberry_train.drop(columns=["id", "yield"])
blueberry_train_y = blueberry_train[["yield"]]

# 교차 검증 설정
kf = KFold(n_splits=15, shuffle=True, random_state=2024)

# n_jobs = -1, cpu에 작업을 각각 할당
def rmse(model):
    score = np.sqrt(-cross_val_score(model, blueberry_train_X, blueberry_train_y, cv=kf,
                                     n_jobs=-1, 
                                     scoring="neg_mean_squared_error").mean())
    return score

# Lasso와 Ridge 모델에 대한 검증 오류 계산
alpha_values = np.logspace(-8, 1, 100) 
lasso_scores = np.zeros(len(alpha_values))
ridge_scores = np.zeros(len(alpha_values))

for i, alpha in enumerate(alpha_values):
    lasso = Lasso(alpha=alpha)
    ridge = Ridge(alpha=alpha)
    lasso_scores[i] = rmse(lasso)
    ridge_scores[i] = rmse(ridge)

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'lasso_validation_error': lasso_scores,
    'ridge_validation_error': ridge_scores
})

# 결과 시각화
# plt.plot(df['lambda'], df['lasso_validation_error'], label='Lasso Validation Error', color='red')
# plt.plot(df['lambda'], df['ridge_validation_error'], label='Ridge Validation Error', color='blue')
# plt.xlabel('Lambda')
# plt.ylabel('Mean Squared Error')
# plt.legend()
# plt.title('Lasso vs Ridge Regression Train vs Validation Error')
# plt.show()

# 최적의 alpha 값 찾기
optimal_lasso_alpha = df['lambda'][np.argmin(df['lasso_validation_error'])]
optimal_ridge_alpha = df['lambda'][np.argmin(df['ridge_validation_error'])]

print("Optimal Lasso lambda:", optimal_lasso_alpha)
print("Optimal Ridge lambda:", optimal_ridge_alpha)

# Optimal Lasso lambda: 0.005336699231206312
# Optimal Ridge lambda: 0.023101297000831626

# 최적의 alpha 값을 가진 Lasso와 Ridge 모델 정의
lasso = Lasso(alpha=optimal_lasso_alpha)
ridge = Ridge(alpha=optimal_ridge_alpha)

# OLS 모델 정의
ols = LinearRegression()

# 모델 학습
lasso.fit(blueberry_train_X, blueberry_train_y)  # 자동으로 기울기, 절편 값을 구해줌
ridge.fit(blueberry_train_X, blueberry_train_y) 
ols.fit(blueberry_train_X, blueberry_train_y)

# 테스트 데이터 준비
blueberry_test_X = blueberry_test.drop(columns=["id"])

total_mean_predict = (lasso.predict(blueberry_test_X) 
                      + ridge.predict(blueberry_test_X) 
                      + ols.predict(blueberry_test_X)) / 3

# 앙상블 예측 결과 저장
sub_df["yield"] = total_mean_predict
sub_df.to_csv("./data/wildblueberry/submission_ensemble_bagging2.csv", index=False) 

# --------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split

## 필요한 데이터 불러오기
blueberry_train = pd.read_csv("./data/wildblueberry/train.csv")
blueberry_test = pd.read_csv("./data/wildblueberry/test.csv")
sub_df = pd.read_csv("./data/wildblueberry/sample_submission.csv")

# train_X, train_y 구분
blueberry_train_X = blueberry_train.drop(columns=["id", "yield"])
blueberry_train_y = blueberry_train[["yield"]]

# 교차 검증 설정
kf = KFold(n_splits=15, shuffle=True, random_state=2024)

# n_jobs = -1, cpu에 작업을 각각 할당
def rmse(model):
    score = np.sqrt(-cross_val_score(model, blueberry_train_X, blueberry_train_y, cv=kf,
                                     n_jobs=-1, 
                                     scoring="neg_mean_squared_error").mean())
    return score

# Lasso와 Ridge 모델에 대한 검증 오류 계산
alpha_values = np.logspace(-8, 1, 100) 
lasso_scores = np.zeros(len(alpha_values))
ridge_scores = np.zeros(len(alpha_values))

for i, alpha in enumerate(alpha_values):
    lasso = Lasso(alpha=alpha)
    ridge = Ridge(alpha=alpha)
    lasso_scores[i] = rmse(lasso)
    ridge_scores[i] = rmse(ridge)

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'lasso_validation_error': lasso_scores,
    'ridge_validation_error': ridge_scores
})

# 결과 시각화
# plt.plot(df['lambda'], df['lasso_validation_error'], label='Lasso Validation Error', color='red')
# plt.plot(df['lambda'], df['ridge_validation_error'], label='Ridge Validation Error', color='blue')
# plt.xlabel('Lambda')
# plt.ylabel('Mean Squared Error')
# plt.legend()
# plt.title('Lasso vs Ridge Regression Train vs Validation Error')
# plt.show()

# 최적의 alpha 값 찾기
optimal_lasso_alpha = df['lambda'][np.argmin(df['lasso_validation_error'])]
optimal_ridge_alpha = df['lambda'][np.argmin(df['ridge_validation_error'])]

print("Optimal Lasso lambda:", optimal_lasso_alpha)
print("Optimal Ridge lambda:", optimal_ridge_alpha)

# Optimal Lasso lambda: 0.005336699231206312
# Optimal Ridge lambda: 0.023101297000831626

# 최적의 alpha 값을 가진 Lasso와 Ridge 모델 정의
lasso = Lasso(alpha=optimal_lasso_alpha)
ridge = Ridge(alpha=optimal_ridge_alpha)

# OLS 모델 정의
ols = LinearRegression()

# 모델 학습
lasso.fit(blueberry_train_X, blueberry_train_y)  # 자동으로 기울기, 절편 값을 구해줌
ridge.fit(blueberry_train_X, blueberry_train_y) 
ols.fit(blueberry_train_X, blueberry_train_y)

# 테스트 데이터 준비
blueberry_test_X = blueberry_test.drop(columns=["id"])

# 모델 예측
ols_predictions = ols.predict(blueberry_test_X)
lasso_predictions = lasso.predict(blueberry_test_X)
ridge_predictions = ridge.predict(blueberry_test_X)

# 예측 성능 평가 (예: 훈련 데이터에서 교차 검증 사용)
ols_rmse = rmse(ols)
lasso_rmse = rmse(lasso)
ridge_rmse = rmse(ridge)

print(f"OLS RMSE: {ols_rmse}")
print(f"Lasso RMSE: {lasso_rmse}")
print(f"Ridge RMSE: {ridge_rmse}")

# 가중 앙상블
lasso_weight = 0.5
ridge_weight = 0.3
ols_weight = 0.2

weighted_ensemble_predictions = (lasso_weight * lasso_predictions +
                                 ridge_weight * ridge_predictions +
                                 ols_weight * ols_predictions)

# 가중 앙상블 예측값 저장
sub_df["yield"] = weighted_ensemble_predictions
sub_df.to_csv("./data/wildblueberry/submission_weighted_ensemble.csv", index=False)

# ------------------------------------------------------------------------------------

