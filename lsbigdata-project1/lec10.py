# 행렬 가지고 놀기

import numpy as np

# 두 개의 벡터를 합쳐 행렬 생성
matrix = np.column_stack((np.arange(1, 5),
                          np.arange(12, 16)))

matrix1 = np.row_stack((np.arange(1, 5),
                       np.arange(12, 16)))

type(matrix)
print("행렬:\n", matrix)

np.zeros(5)
np.zeros((5,4))
np.arange(1,7).reshape((2,3))
# -1 통해서 크기를 자동으로 결정할 수 있음
np.arange(1,7).reshape((2,-1))

# Quiz. 0에서 부터 99까지 수 중 랜덤하게 50개의 숫자를 뽑고,
# 5 by 10 행렬을 만드세요 (정수)
np.random.seed(2024)
a = np.random.randint(0,100,50).reshape((5,10))
a

np.arange(1,21).reshape((4,5), order='F') # 열 우선순서
np.arange(1,21).reshape((4,5), order='C') # 행 우선순서

mat_a = np.arange(1,21).reshape((4,5), order='F')
mat_a

# 인덱싱
mat_a[0,0]
mat_a[1,1]
mat_a[2:3]
mat_a[2,3]
mat_a[0:2,3]
mat_a[1:3, 1:4]

# 행자리, 열자리 비어있는 경우 전체 행, 또는 열 선택
mat_a[3,]
mat_a[3,:]
mat_a[3,::2]

# 짝수 행만 선택하려면?
mat_b = np.arange(1,101).reshape((20,-1))
mat_b[1::2, :]

mat_b[[1,4,6,20],] # 인덱스 범위 error

x = np.arange(1,11).reshape((5,-1)) * 2
x
x[[True,True,False,False,True], 0]

mat_b[:,1]                 # 벡터
mat_b[:,1].reshape((-1,1)) # 행렬
mat_b[:,(1,)]              # 행렬
mat_b[:,[1]]  
mat_b[:,1:2] 

# 필터링(True False 벡터를 사용하여 필터링 할 수 있다.)
# 2번째 열에서 7의 배수가 해당하는 행 알아내는 방법?
mat_b[mat_b[:, 1] % 7 == 0, :]

mat_b[mat_b[:, 1] > 50, :]

# 사진은 행렬이다.
import numpy as np
import matplotlib.pyplot as plt

# 난수 생성하여 3x3 크기의 행렬 생성
np.random.seed(2024)
img1 = np.random.rand(3, 3)
print("이미지 행렬 img1:\n", img1)

# 행렬을 이미지로 표시
plt.imshow(img1, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()

np.random.seed(2024)
a = np.random.randint(0,256,20).reshape(4,-1)
a / 255

plt.imshow(a/255, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()

plt.clf()

# transpose
# 5행 2열의 행렬 생성
x = np.arange(1, 11).reshape((5, 2)) * 2
print("원래 행렬 x:\n", x)

x.transpose()

# 젤리 사진 가져오기
import urllib.request

img_url = "https://bit.ly/3ErnM2Q"
urllib.request.urlretrieve(img_url, "img/jelly.png")

! pip install imageio
import imageio
import numpy as np

# 이미지 읽기
jelly = imageio.imread("img/jelly.png")
print("이미지 클래스:", type(jelly))
print("이미지 차원:", jelly.shape)
print("이미지 첫 4x4 픽셀, 첫 번째 채널:\n", jelly[:4, :4, 0])

len(jelly)
jelly.shape

jelly[:, :, 0]
jelly[:, :, 1]
jelly[:, :, 2]
jelly[:, :, 3]

jelly[:,:,0].shape
jelly[:,:,0].transpose().shape

plt.imshow(jelly)
plt.show()
plt.clf()

# plt.imshow(jelly[:,:,0].transpose())
# plt.imshow(jelly[:,:,0]) # R
# plt.imshow(jelly[:,:,1]) # G
# plt.imshow(jelly[:,:,2]) # B
# plt.imshow(jelly[:,:,3]) # 투명도
# plt.axis('off') # 축 정보 없애기


# 3차원 배열
# 두 개의 2x3 행렬 생성
mat1 = np.arange(1, 7).reshape(2, 3)
mat2 = np.arange(7, 13).reshape(2, 3)

# 3차원 배열로 합치기
my_array = np.array([mat1, mat2])
my_array.shape

first_slice = my_array[0, :, :]
first_slice

filtered_array = my_array[:, :, :-1]
filtered_array

my_array[:, :, [0,2]]
my_array[:, 0, :]
# my_array[0, 1, [1,2]]
my_array[0, 1, 1:3]

mat_x = np.arange(1,101).reshape((5,5,4))
mat_y = np.arange(1,101).reshape((-1,5,2))

mat_z = np.arange(1,101).reshape((-1,3,3))
# Traceback (most recent call last):
#  File "<string>", line 1, in <module>
# ValueError: cannot reshape array of size 100 into shape (3,3)

mat_h = mat_z = np.arange(1,100).reshape((-1,3,3))
mat_h

my_array2 = np.array([my_array, my_array])
my_array2[0, :, :, :]
my_array2.shape

# 넘파이 배열 메서드
a = np.array([[1,2,3],[4,5,6]])

a.sum()
a.sum(axis=0)
a.sum(axis=1)

a.mean()
a.mean(axis=0)
a.mean(axis=1)

mat_b = np.random.randint(1, 100, 50).reshape((5,-1))
mat_b

# 가장 큰 수는?
mat_b.max()

# 행별로 가장 큰 수는?
mat_b.max(axis=1)

# 열별로 가장 큰 수는?
mat_b.max(axis=0)

a = np.array([1,3,2,5]).reshape((2,2))
a
a.cumsum()

# 누적 합
a = np.array([1,3,2,5])
a.cumsum()

# 누적 곱
a = np.array([1,3,2,5])
a.cumprod()

# 행 별 누적 합
mat_b.sum()
mat_b.cumsum(axis = 1)

# flatten()
mat_b.reshape((2,5,5))
mat_b.reshape((2,5,5)).flatten()
mat_b.flatten()

# clip()
d = np.array([1,2,3,4,5])
d.clip(2,4)

# tolist()
d.tolist()
