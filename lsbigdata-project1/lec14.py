import pandas as pd
import numpy as np

old_seat=np.arange(1,29)
?np.random.choice

np.random.seed(20240729)
# 1~28 숫자 중에서 중복 없이 28개 숫자를 뽑는 방법
new_seat= np.random.choice(old_seat, 28, False)

result=pd.DataFrame({
    'old_seat' : old_seat, 'new_seat' : new_seat})

result.to_csv('result.csv')

# 1.
# y = 2x 그래프 그리기
# 점을 직선으로 이어서 표현
import matplotlib.pyplot as plt

x = np.linspace(0, 8, 2)
y = 2 * x
plt.scatter(x,y, s=6)
plt.plot(x, y, color="black")
# plt.plot(x, y, color="black")
plt.show()
plt.clf()

# 2.
# y = x^2를 점 3개 사용해서 그리기
x = np.linspace(-8, 8, 3)
y = x ** 2
plt.scatter(x,y,s=6)
plt.plot(x, y, color="black")
plt.show()
plt.clf()


# y = x^2를 점 100개 사용해서 그리기
x = np.linspace(-8, 8, 100)
y = x ** 2
plt.scatter(x,y,s=6)
plt.plot(x, y, color="black")

# 비율 맞추기
plt.axis('equal')

plt.show()
plt.clf()

# 3.
x = np.linspace(-8, 8, 100)
y = x ** 2
plt.scatter(x,y,s=6)
plt.plot(x, y, color="black")

# x, y 범위 지정
plt.xlim(-10,10)
plt.ylim(0,40)
# 비율 맞추기
# plt.axis('equal')는 xlim, ylim과 같이 사용 X
plt.show()
plt.clf()

# ADP 교재 57p 연습문제
# 신뢰구간 구하기
# 다음은 한 고등학교의 3학년 학생들 중 16명을 무작위로 선별하여 몸무게를 측정한 데이터이다. 
# 이 데이터를 이용하여 해당 고등학교 3학년 전체 남학생들의 몸무게 평균을 예측하고자 한다.
# 79.1, 68.8, 62.0, 74.4, 71.0, 60.6, 98.5, 86.4, 73.0, 40.8, 61.2, 68.7, 61.6, 67.7, 61.7, 66.8
# 단, 해당 고등학교 3학년 남학생들의 몸무게 분포는 정규분포를 따른다고 가정한다.
# 1) 모평균에 대한 95% 신뢰구간을 구하세요.
# 2) 작년 남학생 3학년 전체 분포의 표준편차는 6kg 이었다고 합니다. 이 정보를 이번 년도 남학생
# 분포의 표준편차로 대체하여 모평균에 대한 90% 신뢰구간을 구하세요.

from scipy.stats import norm
import numpy as np

x = np.array([79.1, 68.8, 62.0, 74.4, 71.0, 60.6, 98.5, 86.4, 73.0, 40.8, 61.2, 68.7, 61.6, 67.7, 61.7, 66.8])
x.mean()
len(x)

z_005 = norm.ppf(0.95, loc=0, scale=1)
z_005

# 2) 90% 신뢰구간
x.mean() +  z_005 * 6 / np.sqrt(16)
x.mean() -  z_005 * 6 / np.sqrt(16)

# 데이터로부터 E[X^2] 구하기
x = norm.rvs(loc=3, scale=5, size=10000)

np.mean(x**2)
sum(x**2) / (len(x) - 1)

# E[(x-x**2)/(2*x))] 구하기
x = norm.rvs(loc=3, scale=5, size=10000)
np.mean((x-x**2)/(2*x))

np.random.seed(20240729)
x = norm.rvs(loc=3, scale=5, size=100000)
x_bar = x.mean()
s_2 = sum((x-x_bar)**2)/(len(x)-1)
s_2


# np.var(x) 사용하면 안됨 주의 # n으로 나눈 값
np.var(x, ddof=1) # n-1으로 나눈 값 (표본 분산)

# n-1 vs. n
x = norm.rvs(loc=3, scale=5, size=20)
np.var(x) 
np.var(x, ddof=1)2

# 교재 8장, p212
import pandas as pd
import seaborn as sns

economics = pd.read_csv('data/economics.csv')
economics.head()
economics.info()

sns.lineplot(data=economics, x="date", y="unemploy")
plt.show()
plt.clf()

economics["date2"] = pd.to_datetime(economics['date'])
economics
economics.info()

economics[["date","date2"]]
economics["date2"].dt.year
economics["date2"].dt.month
economics["date2"].dt.day
economics["date2"].dt.month_name()
economics["date2"].dt.quarter
economics["quarter"] = economics["date2"].dt.quarter
economics[["date2","quarter"]]

# 각 날짜는 무슨 요일인가?
economics["date2"].dt.day_name()
economics["date2"] + pd.DateOffset(days=3)
economics["date2"] + pd.DateOffset(days=30)
economics["date2"] + pd.DateOffset(months=1)
economics["date2"].dt.is_leap_year # 윤년 체크

# 연도 변수 추가
economics['year'] = economics["date2"].dt.year
economics.head()

# x축에 연도 표시
sns.lineplot(data=economics, x='year', y='unemploy')
plt.show()
plt.clf()

# 신뢰구간 제거
sns.lineplot(data = economics, x='year', y='unemploy', errorbar = None)
plt.show()
plt.clf()

economics.head(18)

sns.scatterplot(data=economics, x='year', y='unemploy', s=2)

my_df = economics.groupby('year', as_index=False)\
                 .agg(
                         mon_mean = ('unemploy', 'mean'),
                         mon_std = ('unemploy', 'std'),
                         mon_n = ('unemploy', 'count')
                      )
my_df



# mean + 1.96 * std / sqrt(12)
my_df['left_ci'] = my_df['mon_mean'] - 1.96 * my_df['mon_std'] / np.sqrt(my_df['mon_n'])
my_df['right_ci'] = my_df['mon_mean'] + 1.96 * my_df['mon_std'] / np.sqrt(my_df['mon_n'])

import matplotlib.pyplot as plt

x = my_df['year']
y = my_df['mon_mean']
plt.plot(x,y,color='black')
plt.scatter(x,my_df['left_ci'], color = 'green', s=5)
plt.scatter(x,my_df['right_ci'], color = 'green', s=5)
plt.show()
plt.clf()


