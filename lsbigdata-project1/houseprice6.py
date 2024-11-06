# 회귀모델을 통한 집값 예측
# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 필요한 데이터 불러오기
house_train = pd.read_csv('./data/houseprice/train.csv')
house_test = pd.read_csv('./data/houseprice/test.csv')
sub_df = pd.read_csv('./data/houseprice/sample_submission.csv')

# 이상치 탐색
house_train.query("GrLivArea > 4500")
house_train['GrLivArea'].sort_values(ascending=False).head(2)
house_train = house_train.query("GrLivArea <= 4500")

# test의 결측치를 평균값으로 대체
house_test['TotalBsmtSF'].fillna(house_test['TotalBsmtSF'].mean(), inplace=True)
house_test['GarageArea'].fillna(house_test['GarageArea'].mean(), inplace=True)
house_test['GarageCars'].fillna(house_test['GarageCars'].mean(), inplace=True)

# 회귀분석 적합(fit)하기
x = np.array(house_train["GrLivArea"]).reshape(-1,1)
y = house_train["SalePrice"]

#선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y)

# 선형 회귀 모델 생성하면서 동시에 학습 시킬 수 있음
model = LinearRegression().fit(x,y)
model.coef_ # 기울기 a
model.intercept_ # 절편 b

# 회귀 직선의 기울기와 절편
model.coef_ # 기울기 a
model.intercept_ # 절편 b

# 예측값 계산
y_pred = model.predict(x)
y_pred

# 시각화
plt.scatter(x, y, color='blue', label='GrLivArea')
plt.plot(x, y_pred, color='red', label='SalePrice')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([0,5000])
plt.ylim([0,900000])
plt.legend()
plt.show()
plt.clf()

# test_x 녀석 할당하기
test_x = np.array(house_test["GrLivArea"]).reshape(-1,1)
test_x

# test set에 대한 집값
pred_y = model.predict(test_x)
pred_y  

# SalePrice 바꿔치기
sub_df['SalePrice'] = pred_y
sub_df

# csv 파일로 내보내기
# sub_df.to_csv('./data/houseprice/sample_submission17.csv', index=False)
