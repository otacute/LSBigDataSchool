import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform
from sklearn.linear_model import LinearRegression

# 20차 모델 성능을 알아보자
np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

import pandas as pd
df = pd.DataFrame({
    "y" : y,
    "x" : x
})
df

train_df = df.loc[:19]
train_df

for i in range(2, 21):
    train_df[f"x{i}"] = train_df["x"] ** i
    
# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
train_x = train_df[["x"] + [f"x{i}" for i in range(2, 21)]]
train_y = train_df["y"]

valid_df = df.loc[20:]
valid_df

for i in range(2, 21):
    valid_df[f"x{i}"] = valid_df["x"] ** i

# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
valid_x = valid_df[["x"] + [f"x{i}" for i in range(2, 21)]]
valid_x
valid_y = valid_df["y"]
valid_y

from sklearn.linear_model import Lasso

# 결과 받기 위한 벡터 만들기
val_result=np.repeat(0.0, 100)
tr_result=np.repeat(0.0, 100)

for i in np.arange(0, 100):
    model= Lasso(alpha=i*0.01)
    model.fit(train_x, train_y)

    # 모델 성능
    y_hat_train = model.predict(train_x)
    y_hat_val = model.predict(valid_x)

    perf_train=sum((train_df["y"] - y_hat_train)**2)
    perf_val=sum((valid_df["y"] - y_hat_val)**2)
    tr_result[i]=perf_train
    val_result[i]=perf_val

tr_result
val_result

import seaborn as sns

df = pd.DataFrame({
    'lambda': np.arange(0, 1, 0.01), 
    'tr': tr_result,
    'val': val_result
})

# seaborn을 사용하여 산점도 그리기
sns.scatterplot(data=df, x='lambda', y='tr')
sns.scatterplot(data=df, x='lambda', y='val', color='red')
plt.xlim(0, 0.4)

val_result[0]
val_result[1]
np.min(val_result)

# alpha를 0.03로 선택!
np.argmin(val_result)
np.arange(0, 1, 0.01)[np.argmin(val_result)]

# ---------------- 2024.08. 27(화) --------------------

model = Lasso(alpha=0.03)
model.fit(train_x, train_y)
model.coef_
model.intercept_
# model.predict(test_x)

sorted_train=train_x.sort_values("x")
reg_line = model.predict(sorted_train)

plt.plot(sorted_train["x"], reg_line, color="red")
plt.scatter(valid_df["x"], valid_df["y"], color="blue")

# ---------------------------------------------------------
# 추정된 라쏘(lambda=0.03) 모델을 사용해서, 4-, 4 까지 
# 간격 0.01 x에 대하여 예측 값을 계산하고 산점도에 vaild set 그린 다음, 
# -4, 4 까지 예측값을 빨간 선으로 겹쳐서 그림 그리기
model= Lasso(alpha=0.03)
model.fit(train_x, train_y)
model.coef_
model.intercept_
# model.predict(test_x)

k = np.arange(-4,4,0.01)

df_k = pd.DataFrame({
    "x" : k
})

df_k

for i in range(2, 21):
    df_k[f"x{i}"] = df_k["x"] ** i

reg_line=model.predict(df_k)

plt.plot(df_k["x"], reg_line, color="red")
plt.scatter(valid_df["x"], valid_df["y"], color="blue")

# ---------------------------------------------------------
