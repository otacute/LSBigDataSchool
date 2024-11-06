import pandas as pd
import numpy as np

# 변수 1. ExterQual
# ExterQual: Evaluates the quality of the material on the exterior 
# 		
#        Ex	Excellent
#        Gd	Good
#        TA	Average/Typical
#        Fa	Fair
#        Po	Poor

house_train = pd.read_csv('./data/houseprice/train.csv')
house_train.info()
house_train = house_train[["Id", "ExterQual", "SalePrice"]]
house_train.info()

# ExterQual의 고윳값 확인
house_train["ExterQual"].value_counts()

# ExterQual 평균
E_house_mean = house_train.groupby("ExterQual", as_index=False)\
           .agg(mean_E = ("SalePrice", "mean"))
E_house_mean

# ExterQual 평균 시각화
import matplotlib.pyplot as plt
x = E_house_mean["ExterQual"]
y = E_house_mean['mean_E']
plt.plot(x, y)
plt.show()
plt.clf()

# ExterQual의 Ex, Gd, TA, Fa, Po를 값 변경 5, 4, 3, 2, 1
# 고윳값을 숫자로 변환할 매핑 딕셔너리 정의
quality_mapping = {
    'Ex': 5,
    'Gd': 4,
    'TA': 3,
    'Fa': 2
}

# ExterQual 열을 숫자로 변환
house_train["ExterQual"] = house_train["ExterQual"].replace(quality_mapping)

# ExterQual 평균 계산
E_house_mean = house_train.groupby("ExterQual", as_index=False)\
           .agg(mean_E = ("SalePrice", "mean"))

# ExterQual 평균 시각화
x = E_house_mean["ExterQual"]
y = E_house_mean['mean_E']

plt.plot(x, y, marker='o', linestyle='-')

plt.xlabel('ExterQual (Quality Rating)')
plt.ylabel('Mean SalePrice')
plt.title('Mean SalePrice by ExterQual')
plt.xticks(ticks=[2, 3, 4, 5]) # 정수만 표시되게 함
plt.show()
plt.clf()

# 상관계수 correlation을 구함
# correlation : np.float64(0.9819325684393003)
# 1 에 가까울 수록 상관관계가 높음
correlation = E_house_mean["ExterQual"].corr(E_house_mean["mean_E"])

# ---------------------------------------------------------------------

# 변수 2. Fireplaces: Number of fireplaces

house_train = pd.read_csv('./data/houseprice/train.csv')
house_train.info()
house_train = house_train[["Id", "Fireplaces", "SalePrice"]]
house_train.info()

# Fireplaces의 고윳값 확인
house_train["Fireplaces"].value_counts()

# Fireplaces 평균
F_house_mean = house_train.groupby("Fireplaces", as_index=False)\
           .agg(mean_F = ("SalePrice", "mean"))
F_house_mean

# Fireplaces 평균 시각화
import matplotlib.pyplot as plt
x = F_house_mean["Fireplaces"]
y = F_house_mean['mean_F']
plt.plot(x,y)
plt.show()
plt.clf()

# 상관계수 correlation을 구함
# correlation : np.float64(0.937085010309497)
# 1 에 가까울 수록 상관관계가 높음
correlation = F_house_mean["Fireplaces"].corr(F_house_mean["mean_F"])

# ---------------------------------------------------------------------

# 변수 3. OverallQual(값이 높게 나왔었기에 다시 상관계수를 구해보자)
import pandas as pd
import numpy as np

house_train = pd.read_csv('./data/houseprice/train.csv')
house_train.info()
house_train = house_train[["Id", "OverallQual","SalePrice"]]
house_train.info()

# OverallQual 평균
o_house_mean = house_train.groupby("OverallQual", as_index=False)\
           .agg(mean_o = ("SalePrice", "mean"))
o_house_mean

# OverallQual 평균 시각화
import matplotlib.pyplot as plt
x = o_house_mean["OverallQual"]
y = o_house_mean['mean_o']
plt.plot(x, y)
plt.show()
plt.clf()

# 상관계수 correlation을 구함
# correlation : np.float64(0.9572188511562021)
# 1 에 가까울 수록 상관관계가 높음
correlation = o_house_mean["OverallQual"].corr(o_house_mean["mean_o"])

# ---------------------------------------------------------------------

# 변수 4. GrLivArea: Above grade (ground) living area square feet
house_train = pd.read_csv('./data/houseprice/train.csv')
house_train.info()
house_train = house_train[["Id", "GrLivArea", "SalePrice"]]
house_train.info()

# GrLivArea의 고윳값 확인
house_train["GrLivArea"].value_counts()

# GrLivArea 평균
G_house_mean = house_train.groupby("GrLivArea", as_index=False)\
           .agg(mean_G = ("SalePrice", "mean"))
G_house_mean

# GrLivArea 평균 시각화
import matplotlib.pyplot as plt
x = G_house_mean["GrLivArea"]
y = G_house_mean['mean_G']
plt.plot(x, y)
plt.show()
plt.clf()


# 상관계수 correlation을 구함
# correlation : np.float64(0.7288441040721607)
# 1 에 가까울 수록 상관관계가 높음
correlation = G_house_mean["GrLivArea"].corr(G_house_mean["mean_G"])

# ---------------------------------------------------------------------
# 결론 : 제일 높은 상관 계수를 가진 변수는 ExterQual(0.98), OverallQual(0.95)
# ---------------------------------------------------------------------

house_train = pd.read_csv('./data/houseprice/train.csv')

# ExterQual의 Ex, Gd, TA, Fa, Po를 값 변경 5, 4, 3, 2, 1
# 고윳값을 숫자로 변환할 매핑 딕셔너리 정의
quality_mapping = {
    'Ex': 5,
    'Gd': 4,
    'TA': 3,
    'Fa': 2
}

# ExterQual 열을 숫자로 변환
house_train["ExterQual"] = house_train["ExterQual"].replace(quality_mapping)

EO_house_mean= house_train.groupby(["ExterQual","OverallQual"],as_index=False)\
                            .agg(price_mean = ("SalePrice","mean"))

# test 데이터 불러오기
house_test = pd.read_csv('./data/houseprice/test.csv')

# ExterQual의 Ex, Gd, TA, Fa, Po를 값 변경 5, 4, 3, 2, 1
# 고윳값을 숫자로 변환할 매핑 딕셔너리 정의
quality_mapping = {
    'Ex': 5,
    'Gd': 4,
    'TA': 3,
    'Fa': 2
}

# ExterQual 열을 숫자로 변환
house_test["ExterQual"] = house_test["ExterQual"].replace(quality_mapping)
house_test = house_test[["Id", "ExterQual", "OverallQual"]]
house_test

# merge
house_test = pd.merge(house_test, EO_house_mean, how="left", on =["ExterQual", "OverallQual"])
house_test = house_test.rename(columns={'price_mean':"SalePrice"})
house_test

# null값 발생
sum(house_test["SalePrice"].isna())

# 비어 있는 테스트 세트 집들을 확인
house_test.loc[house_test["SalePrice"].isna()]

# 집 값 채우기
house_mean = house_train["SalePrice"].mean()
house_test['SalePrice'] = house_test['SalePrice'].fillna(house_mean)

# sub 데이터 불러오기
sub_df = pd.read_csv('./data/houseprice/sample_submission.csv')
sub_df

# SalePrice 바꿔치기
sub_df['SalePrice'] = house_test['SalePrice']
sub_df

sub_df.to_csv('./data/houseprice/sample_submission4.csv', index=False)

# ---------------------------------------------------------------------
# 결과 제출 시 0.22560 나와서 다시 하나의 변수로 실행, ExterQual(0.98)
# ---------------------------------------------------------------------

# 변수 1. ExterQual만 고려해서 제출해보자.
# ExterQual: Evaluates the quality of the material on the exterior 
# 		
#        Ex	Excellent
#        Gd	Good
#        TA	Average/Typical
#        Fa	Fair
#        Po	Poor

house_train = pd.read_csv('./data/houseprice/train.csv')
house_train.info()
house_train = house_train[["Id", "ExterQual", "SalePrice"]]
house_train.info()

# ExterQual의 Ex, Gd, TA, Fa, Po를 값 변경 5, 4, 3, 2, 1
# 고윳값을 숫자로 변환할 매핑 딕셔너리 정의
quality_mapping = {
    'Ex': 5,
    'Gd': 4,
    'TA': 3,
    'Fa': 2,
    
}

# ExterQual 열을 숫자로 변환
house_train["ExterQual"] = house_train["ExterQual"].replace(quality_mapping)

# ExterQual 평균 계산
E_house_mean = house_train.groupby("ExterQual", as_index=False)\
           .agg(mean_E = ("SalePrice", "mean"))
           
# test 데이터 불러오기
house_test = pd.read_csv('./data/houseprice/test.csv')

# ExterQual의 Ex, Gd, TA, Fa, Po를 값 변경 5, 4, 3, 2, 1
# 고윳값을 숫자로 변환할 매핑 딕셔너리 정의
quality_mapping = {
    'Ex': 5,
    'Gd': 4,
    'TA': 3,
    'Fa': 2
}

# ExterQual 열을 숫자로 변환
house_test["ExterQual"] = house_test["ExterQual"].replace(quality_mapping)
house_test = house_test[["Id", "ExterQual"]]
house_test

# merge
house_test = pd.merge(house_test, E_house_mean, how="left", on = "ExterQual")
house_test = house_test.rename(columns={'mean_E':"SalePrice"})
house_test

sum(house_test["SalePrice"].isna())

# sub 데이터 불러오기
sub_df = pd.read_csv('./data/houseprice/sample_submission.csv')
sub_df

# SalePrice 바꿔치기
sub_df['SalePrice'] = house_test['SalePrice']
sub_df

sub_df.to_csv('./data/houseprice/sample_submission5.csv', index=False)

# ---------------------------------------------------------------------
# 결과 ExterQual(0.98) 하나의 변수로 실행한 결과는 0.30558이 나옴
# 상관관계가 높다고 무조건 SalePrice의 예측값이 높은 결과가 나오는 것이 아닌것이라 추측 중
# ---------------------------------------------------------------------

