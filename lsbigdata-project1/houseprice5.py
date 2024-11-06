# 2024. 08. 01(목) - 회귀분석

# 직선의 방정식
# y = ax + b
# 예) y = 2 x + 3 의 그래프를 그려보세요!

a = 2
b = 3

import numpy as np

x = np.linspace(-5, 5, 100)
y = a * x + b

import matplotlib.pyplot as plt

plt.plot(x, y, color = 'blue')
plt.axvline(0, color = 'black')
plt.axhline(0, color = 'black')
plt.xlim(-5,5)
plt.ylim(-5,5)
plt.show()
plt.clf()

# --------------------------------------------------
import pandas as pd

a = 24
b = 109

x = np.linspace(0, 5, 100)
y = a * x + b 

house_train = pd.read_csv('./data/houseprice/train.csv')
house_train.info()
my_df = house_train[["BedroomAbvGr", "SalePrice"]].head(1000)
my_df["SalePrice"] = my_df["SalePrice"] / 1000
plt.scatter(x=my_df["BedroomAbvGr"], y=my_df["SalePrice"])

mean_bed_room = my_df.groupby("BedroomAbvGr",  as_index=False)\
                    .agg(mean_bedroom = ("SalePrice", "mean"))
mean_bed_room

my_df["BedroomAbvGr"].value_counts()

plt.plot(x, y, color = 'blue')
plt.xlim(-1,7)
plt.show()
plt.clf()
# --------------------------------------------------

# test 데이터 불러오기
house_test = pd.read_csv('./data/houseprice/test.csv')
house_test = house_test[["Id", "BedroomAbvGr"]]
house_test

a = 36
b = 68
x = house_test["BedroomAbvGr"]
y = a * x + b 

house_test["SalePrice"] = y * 1000
house_test

# sub 데이터 불러오기
sub_df = pd.read_csv('./data/houseprice/sample_submission.csv')
sub_df

# SalePrice 바꿔치기
sub_df['SalePrice'] = house_test['SalePrice']
sub_df

sub_df.to_csv('./data/houseprice/sample_submission7.csv', index=False)

# issac 센세 --------------------------------------------------
# test 데이터 불러오기
house_test = pd.read_csv('./data/houseprice/test.csv')

# sub 데이터 불러오기
sub_df = pd.read_csv('./data/houseprice/sample_submission.csv')
sub_df

a = 70; b=10
sub_df["SalePrice"] = (a * house_test["BedroomAbvGr"] + b) * 1000
sub_df

# --------------------------------------------------
# 직선 성능 평가
a = 36
b = 68

# y_hat 은 어떻게 구할까
house_train = pd.read_csv('./data/houseprice/train.csv')
y_hat = (a * house_train['BedroomAbvGr'] + b) * 1000

# y는 어디에 있는가
y = house_train['SalePrice']
np.abs(y - y_hat) # 절대거리
np.sum(np.abs(y - y_hat)) # 절대값 합
np.sum((y - y_hat)**2) # 제곱합

# --------------------------------------------------

# !pip install scikit-learn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 예시 데이터 (x와 y 벡터)
x = np.array([1, 3, 2, 1, 5]).reshape(-1, 1)  # x 벡터 (특성 벡터는 2차원 배열이어야 합니다)
y = np.array([1, 2, 3, 4, 5])  # y 벡터 (레이블 벡터는 1차원 배열입니다)

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_ # 기울기 a
model.intercept_ # 절편 b
slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope): {slope}")
print(f"절편 (intercept): {intercept}")

# 예측값 계산
y_pred = model.predict(x)

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='data')
plt.plot(x, y_pred, color='red', label='regression line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()

# --------------------------------------------------
# 회귀모델을 통한 집값 예측
# 필요한 패키지 불러오기
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 필요한 데이터 불러오기
house_train = pd.read_csv('./data/houseprice/train.csv')
house_test = pd.read_csv('./data/houseprice/test.csv')
sub_df = pd.read_csv('./data/houseprice/sample_submission.csv')

# 회귀분석 적합(fit)하기
x = np.array(house_train["BedroomAbvGr"]).reshape(-1,1)
y = house_train["SalePrice"] / 1000

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

plt.scatter(x, y, color='blue', label='BedroomAbvGr')
plt.plot(x, y_pred, color='red', label='SalePrice')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()

# --------------------------------------------------

import numpy as np
from scipy.optimize import minimize

# 최소값을 찾을 다변수 함수 정의
def my_f(x):
    return (x[0] - 1) ** 2 + (x[1] - 2) ** 2

# 초기 추정값
initial_guess = [0, 0]

# 최소값 찾기
result = minimize(my_f, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)


# 회귀직선 구하기

import numpy as np
from scipy.optimize import minimize

def line_perform(par):
    y_hat=(par[0] * house_train["BedroomAbvGr"] + par[1]) * 1000
    y=house_train["SalePrice"]
    return np.sum(np.abs((y-y_hat)))

line_perform([36, 68])

# 초기 추정값
initial_guess = [0, 0]

# 최소값 찾기
result = minimize(line_perform, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x) 

# --------------------------------------------------
# 2024.08.02(금)

# y = x^2 + 3의 최소값이 나오는 입력값 구하기
def my_f(x) : 
    return x ** 2 + 3

my_f(3)

import numpy as np
from scipy.optimize import minimize

# 초기 추정값
initial_guess = [10]

# 최소값 찾기
result = minimize(my_f, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x) 

# --------------------------------------------------
# z = x^2 + y^2 + 3
def my_f2(x) : 
    return x[0] ** 2 + x[1] ** 2 + 3

my_f2([1,3])

import numpy as np
from scipy.optimize import minimize

# 초기 추정값
initial_guess = [0,0]

# 최소값 찾기
result = minimize(my_f2, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x) 
# --------------------------------------------------
# f(x,y,z) = (x-1)^2 + (y-2)^2 + (z-4)^2 + 7
def my_f3(x) : 
    return (x[0] - 1) ** 2 + (x[1] - 2) ** 2 + (x[2] - 4) ** 2 + 7

import numpy as np
from scipy.optimize import minimize

# 초기 추정값
initial_guess = [0,0,0]

# 최소값 찾기
result = minimize(my_f3, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x) 
# --------------------------------------------------
# test 데이터 set 불러오기
house_test = pd.read_csv('./data/houseprice/test.csv')

test_x = np.array(house_test["BedroomAbvGr"]).reshape(-1,1)
test_x

# test set에 대한 집값
pred_y = model.predict(test_x)
pred_y  

# SalePrice 바꿔치기
sub_df['SalePrice'] = pred_y * 1000
sub_df

# csv 파일로 내보내기
sub_df.to_csv('./data/houseprice/sample_submission8.csv', index=False)

