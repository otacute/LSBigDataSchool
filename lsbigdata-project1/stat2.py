import numpy as np
import matplotlib.pyplot as plt

# 예제 넘파이 배열 생성
# 0~1 사이의 숫자를 10000개 뽑아서 hist 그리기
data = np.random.rand(10000)

# 히스토그램 그리기
plt.hist(data, bins=30, alpha=0.7, color='blue')
# bins = 막대의 너비 조절, alpha = 투명도
plt.title('Histogram of Numpy Vector')
plt.xlabel('Value')
plt.ylabel('Frequency')
# 격자 표시 유무
plt.grid(True)
plt.show()
plt.clf()

# 1. 0~1 사이 숫자 5개 발생
# 2. 표본 평균 계산하기
# 3. 1,2단계를 10000번 반복해서 벡터로 만들기
# 4. 히스토그램 그리기

# x = np.random.rand(50000).reshape(-1,5).mean(axis=1)
# reshape(-1, 5) 5번 뽑는 것을 10000번 반복하게 만듬

x = np.random.rand(10000,5).mean(axis=1)
# rand 값을 추출할 때 (10000, 5)의 형태로 만든 후 열 방향으로 mean을 계산

plt.hist(x, bins=30, alpha=0.7, color='blue')
plt.title('Histogram of Numpy Vector')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
plt.clf()

# ----- 2024.07.23(화) 오후 진도 ---------

import numpy as np

x = np.arange(33)
x.sum()/33

np.unique((x - 16)**2)
sum(np.unique((x - 16)**2) * (2/33))

# E[X^2]
sum(x**2 * (1/33))

# Var(X) = E[X^2] - (E[X])^2
sum(x**2 * (1/33)) - 16**2


## Example
x = np.arange(4)
x
pro_x = np.array([1/6,2/6,2/6,1/6])
pro_x

# 기대값
Ex = sum(x *pro_x)
Exx = sum((x ** 2) * pro_x)

# 분산
Exx - Ex**2
sum((x-Ex) **2 * pro_x)

sum(np.arange(50))
sum(np.arange(51))

# quiz2
x = np.arange(99)
x

# 1-50-1 벡터
x_1_50_1 = np.concatenate((np.arange(1,51), np.arange(49, 0, -1)))
pro_x = x_1_50_1/2500

# 기대값
Ex = sum(x * pro_x)
Exx = sum(x**2 * pro_x)

# 분산
Exx - Ex**2
sum((x-Ex)**2 * pro_x)


# Quiz 3
# y = np.arange(0,7,2)
y = np.arange(4) * 2
pro_y = np.array([1/6,2/6,2/6,1/6])

# 기대값
Ey = sum(y *pro_y)
Eyy = sum((y ** 2) * pro_y)

# 분산
Eyy - Ey**2
sum((y-Ey) **2 * pro_y)


9.52 **2 / 16
np.sqrt(9.52 **2 / 16)

9.52 **2 / 10
np.sqrt(9.52 **2 / 10)

np.sqrt(4/3)

np.sqrt(81 /25)

