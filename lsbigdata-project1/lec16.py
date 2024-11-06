# X ~N(3, 7^2)

from scipy.stats import norm

x = norm.ppf(0.25, loc=3, scale=7)
x
z = norm.ppf(0.25, loc=0, scale=1)
z

3 + z * 7

norm.cdf(5, loc=3, scale=7)
norm.cdf(2/7, loc=0, scale=1)

norm.ppf(0.975, loc=0, scale=1)
norm.ppf(0.025, loc=0, scale=1)

# 표본 1000개, 히스토그램, pdf 겹쳐 그리기
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

z = norm.rvs(loc=0, scale=1, size=1000)
z

sns.histplot(z, stat="density", color = 'grey')

# Plot the normal distribution PDF
zmin, zmax = (z.min(), z.max())
z_values = np.linspace(zmin, zmax, 100)
pdf_values = norm.pdf(z_values, loc=0, scale=1)
plt.plot(z_values, pdf_values, color='red', linewidth=2)

plt.show()
plt.clf()

# z와 x 두개의 pdf 겹쳐 그리기
z = norm.rvs(loc=0, scale=1, size=1000)
z
x = (np.sqrt(2) * z) + 3
x

sns.histplot(z, stat="density", color='grey')
sns.histplot(x, stat="density", color='green')

zmin, zmax = (z.min(), z.max())
z_values = np.linspace(zmin, zmax, 100)
z_pdf_values = norm.pdf(z_values, loc=0, scale=1)

xmin, xmax = (x.min(), x.max())
x_values = np.linspace(xmin, xmax, 100)
x_pdf_values = norm.pdf(x_values, loc=3, scale=np.sqrt(2))

plt.plot(z_values, z_pdf_values, color='red', linewidth=2)
plt.plot(x_values, x_pdf_values, color='blue', linewidth=2)
plt.show()
plt.clf()

# 표준화 확인
# X ~ N(3, 5^2) , Z
x = norm.rvs(loc=5, scale=3, size=1000)
x

# 표준화
z = (x - 5) / 3
z
sns.histplot(z, stat="density", color='grey')

# Plot the normal distiribution PDF
zmin, zmax = (z.min(), z.max())
z_values = np.linspace(zmin, zmax, 100)
z_pdf_values = norm.pdf(z_values, loc=0, scale=1)

plt.plot(z_values, z_pdf_values, color='red', linewidth=2)
plt.show()
plt.clf()

# 나랑 한렬이가 짠 코드
# 1. X 표본을 10개 뽑아서 표본 분산값 계산하기
x_10 = norm.rvs(loc=5, scale=3, size=10)
x_10

x_10_var = np.var(x_10)
x_10_var

# 2. X 표본 1000개 뽑음
x_1000 = norm.rvs(loc=5, scale=3, size=1000)
x_1000

# 3. 1번에서 계산한 s^2으로 sigma^2 대체한 표준화를 진행
z = (x_1000 - 5) / np.sqrt(x_10_var)
z

# 4. z의 히스토그램 그리기
sns.histplot(z, stat="density", color='grey')

# Plot the normal distiribution PDF
zmin, zmax = (z.min(), z.max())
z_values = np.linspace(zmin, zmax, 100)
z_pdf_values = norm.pdf(z_values, loc=0, scale=1)

plt.plot(z_values, z_pdf_values, color='red', linewidth=2)
plt.show()
plt.clf()

# issac toast 센세 코드
# 표본표준편차 나눠도 표준 정규분포가 될까?
#1.
x = norm.rvs(loc=5, scale=3, size=10)
s = np.std(x, ddof=1)
# s_2 = np.var(x, ddof=1)
s**2

#2.
x=norm.rvs(loc=5, scale=3, size=1000)

# 표준화
z = (x-5) / s
# z = (x-5) / 3
sns.histplot(z, stat="density", color='grey')

# Plot the normal distiribution PDF
zmin, zmax = (z.min(), z.max())
z_values = np.linspace(zmin, zmax, 100)
z_pdf_values = norm.pdf(z_values, loc=0, scale=1)

plt.plot(z_values, z_pdf_values, color='red', linewidth=2)
plt.show()
plt.clf()

# t 분포에 대해서 알아보자
# X ~ t(df)
# 종모양, 대칭분포, 중심 0
# 모수 df : 자유도라고 부름 - 퍼짐을 나타내는 모수
# df 이 작으면 분산 커짐.
# df 이 무한대로 가면 표준정규분포가 된다.
?t.pdf

from scipy.stats import t

# t.pdf
# t.ppf
# t.cdf
# t.rvs
# 자유도가 4인 t분포의 pdf를 그려보세요!
t_values = np.linspace(-4, 4, 100)
pdf_values = t.pdf(t_values, df=4)
plt.plot(t_values, pdf_values, color='red', linewidth=2)

# 표준정규분포 겹치기
pdf_values = norm.pdf(t_values, loc=0, scale=1)
plt.plot(t_values, pdf_values, color='black', linewidth=2)

plt.show()
plt.clf()

# X ~ ?(mu, sigma^2)
# X bar ~ N(mu, sigma^2/n)
# X bar ~= t(x_bar, s^2/n) 자유도가 n-1인 t분포
x = norm.rvs(loc=15, scale=3, size=16, random_state=42)
x
x_bar = x.mean()
n = len(x)

# df = degree of freedom
# 모분산을 모를 때 : 모평균에 대한 95% 신뢰구간을 구해보자!
x_bar + t.ppf(0.975, df=n-1) * np.std(x, ddof=1) / np.sqrt(n)
x_bar - t.ppf(0.975, df=n-1) * np.std(x, ddof=1) / np.sqrt(n)

# 모분산(3^2)을 알 때 : 모평균에 대한 95% 신뢰구간을 구해보자!
x_bar + norm.ppf(0.975, loc=0, scale=1) * 3 / np.sqrt(n)
x_bar - norm.ppf(0.975, loc=0, scale=1) * 3 / np.sqrt(n)
