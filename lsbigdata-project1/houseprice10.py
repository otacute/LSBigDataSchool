# 2024. 08. 05(월) 수업 --------------------------
# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 필요한 데이터 불러오기
house_train=pd.read_csv("./data/houseprice/train.csv")
house_test=pd.read_csv("./data/houseprice/test.csv")
sub_df=pd.read_csv("./data/houseprice/sample_submission.csv")

house_train.info()

# 이상치 탐색
# house_train=house_train.query("GrLivArea <= 4500")

## 회귀분석 적합(fit)하기
# int, float 숫자형 데이터만 선택하기
x = house_train.select_dtypes(include=[int, float])
x.info()
# 불필요 컬럼 제거하기(Id, SalePrice)
x = x.iloc[:, 1:-1]
x
y = house_train["SalePrice"]

# nan 값 확인
x.isna().sum()
# LotFrontage      259 - 실수값
# MasVnrArea         8 - 벽면을 덮고 있는 벽돌이나 돌 베니어의 총 면적
# GarageYrBlt       81 - 차고가 지어진 년도

# 변수별로 결측값 채우기 - mean과 mode사용
# fill_values = {
#     'LotFrontage' : x["LotFrontage"].mean(),
#     'MasVnrArea' : x["MasVnrArea"].mode()[0],
#     'GarageYrBlt' : x["GarageYrBlt"].mode()[0]
# }
# fill_values

# 변수별로 결측값 채우기 - mean이용
fill_values = {
    'LotFrontage' : x["LotFrontage"].mean(),
    'MasVnrArea' : x["MasVnrArea"].mean(),
    'GarageYrBlt' : x["GarageYrBlt"].mean()
}
fill_values

x = x.fillna(value=fill_values)
x.isna().sum()

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

# test 데이터 예측
test_x = house_test.select_dtypes(include=[int, float])
test_x = test_x.iloc[:, 1:] # test에는 SalePrice가 없으므로 Id만 제거함

# 변수별로 결측값 채우기 - mean과 mode사용
# fill_values = {
#     'LotFrontage' : test_x["LotFrontage"].mean(),
#     'MasVnrArea' : test_x["MasVnrArea"].mode()[0],
#     'GarageYrBlt' : test_x["GarageYrBlt"].mode()[0]
# }
# fill_values

# 변수별로 결측값 채우기 - mean이용
# fill_values = {
#    'LotFrontage' : test_x["LotFrontage"].mean(),
#    'MasVnrArea' : test_x["MasVnrArea"].mean(),
#    'GarageYrBlt' : test_x["GarageYrBlt"].mean()
# }
# fill_values

# test_x = test_x.fillna(value=fill_values)
# test_x.isna().sum()

test_x = test_x.fillna(test_x.mean())

# 결측치 확인
test_x.isna().sum()

# 테스트 데이터 집값 예측
pred_y=model.predict(test_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
# sample_submission19.csv - GrLivArea <= 4500 이상치 제거 후, 
# fill_values = {
#     'LotFrontage' : x["LotFrontage"].mean(),
#     'MasVnrArea' : x["MasVnrArea"].mode()[0],
#     'GarageYrBlt' : x["GarageYrBlt"].mode()[0]
# } 결측치 대체 후 모든 숫자형 변수를 다항회귀 모델을 돌린 SalePrice의 값

# sample_submission20.csv -  
# fill_values = {
#     'LotFrontage' : test_x["LotFrontage"].mean(),
#     'MasVnrArea' : test_x["MasVnrArea"].mean(),
#     'GarageYrBlt' : test_x["GarageYrBlt"].mean()
# } 결측치 대체 후 모든 숫자형 변수를 다항회귀 모델을 돌린 SalePrice의 값

# sub_df.to_csv("./data/houseprice/sample_submission20.csv", index=False)

