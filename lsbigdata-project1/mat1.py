import numpy as np

# 벡터 * 벡터(내적)
a = np.arange(1,4)
b = np.array([3,6,9])

a.dot(b)

# 벡터
a = np.array([1,2,3,4]).reshape((2,2), order='F')
a

b = np.array([5,6]).reshape(2, 1)
b

a.dot(b)
a @ b

# 행렬 * 행렬
a = np.array([1,2,3,4]).reshape((2,2), order='F')
a

b = np.array([5,6,7,8]).reshape((2,2), order='F')
b

a @ b

# Q1
a = np.array([1,2,1,0,2,3]).reshape(2,3)
a

b = np.array([1,0,-1,1,2,3]).reshape(3,2)
b

a @ b

# Q2.
a = np.array([3,5,7,2,4,9,3,1,0]).reshape(3,3)
a @ np.eye(3) 
np.eye(3) @ a

# transpose
a
a.transpose()

b = a[:, 0:2]
b
b.transpose()

# 회귀분석 데이터 행렬
x = np.array([13, 15,
          12, 14,
          10, 11,
          5, 6]).reshape(4, 2)
x
vec1 = np.repeat(1,4).reshape(4,1)
matX = np.hstack((vec1, x))
matX

# case1
beta_vec = np.array([2,3,1]).reshape(3,1)
beta_vec

matX @ beta_vec

# case2
beta_vec = np.array([2,0,1]).reshape(3,1)
beta_vec

matX @ beta_vec

y = np.array([20, 19, 20, 12]).reshape(4,1)
(y - matX @ beta_vec).transpose() @ (y - matX @ beta_vec)

# Inverse matrix
a = np.array([1,5,3,4]).reshape(2,2)
a_inv = (-1/11) * np.array([4, -5, -3, 1]).reshape(2,2)
a @ a_inv

# 3 by 3 역행렬
a = np.array([-4, -6, 2,
              5, -1, 3,
              -2, 4, -3]).reshape(3,3)
a_inv = np.linalg.inv(a)
np.linalg.det(a)
a_inv

np.round(a @ a_inv, 3)

# 역행렬이 존재하지 않는 경우(선형종속)
b = np.array([1, 2, 3,
              2, 4, 5,
              3, 6, 7]).reshape(3,3)
b_inv = np.linalg.inv(b) # Singular matrix 에러남
np.linalg.det(b) # 행렬식이 항상0

# 벡터 형태로 베타 구하기
XtX_inv = np.linalg.inv((matX.transpose() @ matX))
Xty = matX.transpose() @ y
beta_hat = XtX_inv @ Xty
beta_hat

# 모델 fit으로 베타 구하기
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(matX[:, 1:], y)

model.coef_
model.intercept_

# minimize로 베타 구하기
from scipy.optimize import minimize

def line_perform(beta):
    beta = np.array(beta).reshape(3,1)
    a = (y - matX @ beta)
    return (a.transpose() @ a)

line_perform([8.55, 5.96, -4.38])

# 초기 추정값
initial_guess = [0, 0, 0]

# 최소값 찾기
result = minimize(line_perform, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

# minimize로 Lasso베타 구하기
from scipy.optimize import minimize

def line_perform_lasso(beta):
    beta = np.array(beta).reshape(3,1)
    a = (y - matX @ beta)
    return (a.transpose() @ a) + 3 * np.abs(beta).sum()

line_perform([8.55, 5.96, -4.38])
line_perform([3.76, 1.36, 0])
line_perform_lasso([8.55, 5.96, -4.38])
line_perform_lasso([3.76, 1.36, 0])

# 초기 추정값
initial_guess = [0, 0, 0]

# 최소값 찾기
result = minimize(line_perform_lasso, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

# minimize로 Ridge베타 구하기
from scipy.optimize import minimize

def line_perform_ridge(beta):
    beta = np.array(beta).reshape(3,1)
    a = (y - matX @ beta)
    return (a.transpose() @ a) + 3 * (beta**2).sum()

line_perform_ridge([8.55, 5.96, -4.38])
line_perform_ridge([3.76, 1.36, 0])

# 초기 추정값
initial_guess = [0, 0, 0]

# 최소값 찾기
result = minimize(line_perform_ridge, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)


