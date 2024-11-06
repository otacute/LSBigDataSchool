# y = 2 * x + 3 그래프 그리기
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# x 값의 범위 설정 - x는 집의 크기
x = np.linspace(0, 100, 400)

# y값 계산
y = 2 * x + 3

# np.random.seed(20240805)
obs_x = np.random.choice(np.arange(100), 20)
epsilon_i = norm.rvs(loc=0, scale=10, size=20)
obs_y = 2 * obs_x + 3 + epsilon_i

# 그래프 그리기
plt.plot(x, y, label='y = 2x + 3', color='black')
plt.scatter(obs_x, obs_y, color="blue", s=3)
# plt.show()
# plt.clf()

import pandas as pd
df = pd.DataFrame({
    "x" : obs_x,
    "y" : obs_y
})
df

# obs_x, obs_y 사용해서 회귀분석
from sklearn.linear_model import LinearRegression

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
obs_x = obs_x.reshape(-1,1)
model.fit(obs_x, obs_y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_[0]      # 기울기 a_hat
model.intercept_ # 절편 b_hat

# 빨간색 선으로 회귀 직선 그리기
x = np.linspace(0, 100, 400)
y = model.coef_[0] * x + model.intercept_ 
plt.xlim([0,100])
plt.ylim([0,300])
plt.plot(x, y, color="red") # 회귀 직선
plt.show()
plt.clf()

# ! pip install statsmodels
import statsmodels.api as sm
obs_x = sm.add_constant(obs_x)
model = sm.OLS(obs_y, obs_x).fit()
print(model.summary())

np.sqrt(8.79 **2 / 20)

# 유의확률, p-value
# p-value가 작고 크고를 판단하는 기준은 유의수준에 따라 결정됨 
(1 - norm.cdf(18, loc=10, scale=1.96))

# P(X>=18)
1 - norm.cdf(18, loc=10, scale=1.96)

z = (18 - 10) / 1.96
z
1 - norm.cdf(z, loc=0, scale=1)

# 교재 57p
# 신형 자동차의 에너지 소비효율 등급
# 슬통 자동자는 매해 출시되는 신형 자동차의 에너지 소비효율 등급을 1등급으로 유지하고 있다. 
# 22년 개발된 신형 모델이 한국 자동차 평가원에서 설정한 에너지 소비 효율등급 1등급을 받을 수 있을지 검정하려한다. 
# 평가원에 따르면 1등급의 기준은 평균 복합 에너지 소비효율이 16.0 이상인 경우 부여한다고 한다.
# 다음은 신형 자동차 15대의 복합 에너지소비효율 측정한 결과이다.
# 15.078, 15.752, 15.549, 15.56, 16.098, 13.277, 15.462, 16.116, 15.214, 16.93, 14.118, 14.927,15.382, 16.709, 16.804
# 표본에 의하여 판단해볼때, 현대자동차의 신형 모델은 에너지 효율 1등급으로 판단할 수 있을지 판단해보시오. (유의수준 1%로 설정)

# 2. 검정을 위한 가설을 명확하게 서술하시오.
# 귀무가설 H0 : mu >= 16
# 대립가설 HA : mu < 16
# m0 = 16

# 3. 검정통계량 계산하시오.
import numpy as np

energy = [15.078, 15.752, 15.549, 15.56, 16.098, 13.277, 15.462, 16.116, 15.214, 16.93, 14.118, 14.927,15.382, 16.709, 16.804]

# 표본 표준 편차
s = np.std(energy, ddof=1)
s
# 표본 평균
x_bar = np.mean(energy)
# x_bar = sum(energy) / len(energy)
x_bar
# 표본의 개수
n = len(energy)
# 귀무가설 m0
m0 = 16
# t 분포를 따르는 표준화
T = (x_bar - m0) / (s/np.sqrt(n))  
T

# 4. p‑value을 구하세요.
from scipy.stats import t
df = len(energy) - 1

p_values = t.cdf(T, df)
p_values

# 6. 현대자동차의 신형 모델의 평균 복합 에너지 소비효율에 대하여 95% 신뢰구간을 구해보세요.
CI_r = x_bar + t.ppf(0.975, df) * (s/np.sqrt(n))
CI_l = x_bar + t.ppf(0.025, df) * (s/np.sqrt(n))
CI = (CI_l, CI_r)
CI


