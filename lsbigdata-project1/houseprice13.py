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
house_train=pd.read_csv("./data/houseprice/train.csv")
house_test=pd.read_csv("./data/houseprice/test.csv")
sub_df=pd.read_csv("./data/houseprice/sample_submission.csv")


# 통합 df 생성 후 전처리
df = pd.concat([house_train, house_test], ignore_index=True)

X = df.drop(columns=["Id", "SalePrice"])
y = df[["SalePrice"]]

# 각 숫자변수는 평균 채우기
quantitative = X.select_dtypes(include=[int,float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum()>0]

for col in quant_selected :
    X[col].fillna(df[col].mean(), inplace=True)
# 숫자형 변수에 null값 갯수 확인 
X[quant_selected].isna().sum().sum()

# 각 범주형 변수는 Unknown채우기
qualitative = X.select_dtypes(include="object")
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum()>0]

for col in qual_selected :
    X[col].fillna("unknown", inplace=True)
# 범주형 변수에 null값 갯수 확인 
X[qual_selected].isna().sum().sum()

# X(train+test) 더미코딩
X = pd.get_dummies(
    X,
    columns= house_train.select_dtypes(include=[object]).columns,
    drop_first=True
    )
X

# X에 대한 전처리 후 X + y를 통해 다시 합친 후, train과 test 분리
df = pd.concat([X, y], axis=1)
df

train_n = len(house_train)

# train / test 데이터셋
train_df=df.iloc[:train_n,]
test_df=df.iloc[train_n:,]

# train_x, train_y 구분 
train_df_x = train_df.drop(columns="SalePrice")
train_df_y = train_df["SalePrice"]
test_df_x = test_df.drop(columns="SalePrice")

# 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=2024)

# n_jobs = -1, cpu에 작업을 각각 할당
def rmse(model):
    score = np.sqrt(-cross_val_score(model, train_df_x, train_df_y, cv = kf,
                                     n_jobs = -1, 
                                     scoring = "neg_mean_squared_error").mean())
    return(score)

lasso = Lasso(alpha=0.01)
# ridge = Ridge(alpha=0.01)
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

# 최적의 alpha 값 찾기 # 190
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)

# 선형 회귀 모델 생성
model = Lasso(alpha=190)

# 모델 학습
model.fit(train_df_x, train_df_y)  # 자동으로 기울기, 절편 값을 구해줌

pred_y=model.predict(test_df_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
# sub_df.to_csv("./data/houseprice/sample_submission23.csv", index=False)