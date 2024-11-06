from scipy.stats import uniform

uniform.rvs(loc = 2, scale = 4, size=1)

import matplotlib.pyplot as plt
# 내가 짠 코드
plt.plot(np.linspace(0,8,10000), uniform.pdf(np.linspace(0,8,10000), loc=2, scale=4))
plt.show()
plt.clf()

# issac 선생님이 짠 코드
k = np.linspace(0, 8, 100)
y = uniform.pdf(k, loc=2, scale=4)
plt.plot(k, y, color="black")
plt.show()
plt.clf()

# X ~ 균일분포 U(a,b)
# loc : a, scale: b-a
uniform.cdf(3.25, loc=2, scale=4)
uniform.cdf(8.39, loc=2, scale=4) - uniform.cdf(5, loc=2, scale=4)

uniform.ppf(0.93,loc=2, scale=4)

# 표본 20개를 뽑고 표본평균을 구하시오.

x = uniform.rvs(loc=2, scale=4, size=20*1000, 
                random_state=42)
x = x.reshape(1000,20)
x.shape
blue_x = x.mean(axis=1)
blue_x

import seaborn as sns
sns.histplot(blue_x, stat="density")
plt.show()

from scipy.stats import norm
uniform.var(loc=2,  scale=4)
uniform.expect(loc=2, scale=4)

# X bar ~ N(mu, sigma^2/n)
# X bar ~ N(4, 1.333333333/20)

# Plot the normal distribution PDF
xmin, xmax = (blue_x.min(), blue_x.max())
x_values = np.linspace(xmin, xmax, 100)
pdf_values = norm.pdf(x_values, loc=4, 
             scale= np.sqrt(1.333333333/20))
plt.plot(x_values, pdf_values, color='red', linewidth = 2)

plt.show()
plt.clf()

# 신뢰구간
# X bar ~ N(mu, sigma^2/n)
# X bar ~ N(4, 1.333333333/20)
from scipy.stats import norm

# Plot the normal distribution PDF
x_values = np.linspace(3, 5, 100)
pdf_values = norm.pdf(x_values, loc=4, 
             scale= np.sqrt(1.333333333/20))
plt.plot(x_values, pdf_values, color='red', linewidth = 2)

# 표본평균(파란벽돌) 점 찍기
blue_x = uniform.rvs(loc=2, scale=4, size = 20).mean()
# norm.ppf(0.975, loc=0, scale=1) == 1.96

a = blue_x + 0.665
# a = blue_x + 2.57 * np.sqrt(1.333333333/20))
b = blue_x - 0.665
# b = blue_x - 2.57 * np.sqrt(1.333333333/20))
plt.scatter(blue_x, 0.002, 
            color='blue', zorder=10, s=10)
plt.axvline(x=a, color='blue',
            linestyle='--', linewidth=1)
plt.axvline(x=b, color='blue',
            linestyle='--', linewidth=1)

# 기대값 표현
plt.axvline(x=4, color='green',
            linestyle='-', linewidth=2)

plt.show()
plt.clf()

# 95%를 커버 a, b 표준편차 기준 몇 배를 벌리면 되나요?
(4 - norm.ppf(0.025, loc=4, scale= np.sqrt(1.333333333/20))) / np.sqrt(1.333333333/20)

# 99%를 커버 a, b 표준편차 기준 몇 배를 벌리면 되나요?
(4 - norm.ppf(0.005, loc=4, scale= np.sqrt(1.333333333/20))) / np.sqrt(1.333333333/20)


norm.ppf(0.025, loc=4, scale= np.sqrt(1.333333333/20))
norm.ppf(0.975, loc=4, scale= np.sqrt(1.333333333/20))

norm.ppf(0.005, loc=4, scale= np.sqrt(1.333333333/20))
norm.ppf(0.995, loc=4, scale= np.sqrt(1.333333333/20))
