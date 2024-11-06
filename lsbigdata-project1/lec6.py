import numpy as np

# 벡터 슬라이싱 예제, a를 랜덤하게 채움
np.random.seed(2024)
a = np.random.randint(1,21,10)

# rand함수는 시물레이션 시 많이 사용
# seed -> 내가 시물레이션 한 것을 남들에게 보여줄 때 사용

# 함수 사용법 할때
# ?np.random.randint

print(a)

# 두 번째 값 추출
print(a[1])

a[2:5]
a[-2] # 맨 끝에서 두 번째
a[::2] # 처음부터 끝까지 스텝은 2
a[0:6:2]

# 1에서부터 1000사이 3의 배수 합은?
sum(np.arange(3, 1001, 3))
x = np.arange(0, 1001)
sum(x[::3])

print(a[[0,2,4]])

print(np.delete(a,1))

np.delete(a, [1,3])

# False, True 내가 가진 vector들을 필터링 할 때 사용
a > 3
b = a[a>3]
print(b)

np.random.seed(2024)
a = np.random.randint(1,10000, 5)
# a[조건을 만족하는 논리형 벡터]
a[(a>2000) & (a<5000)]

b = a[(a>2) & (a<9)]
print(b)

!pip install pydataset
import pydataset

df=pydataset.data('mtcars')
np_df=np.array(df['mpg'])

model_names = np.array(df.index)
model_names

# 15이상 25이하인 데이터 개수는?
sum((np_df>=15) & (np_df<=25))

# 15이상 20이하인 자동차 모델은?
model_names[(np_df>=15) & (np_df<=20)]

# 평균 mpg 보다 높은(이상) 자동차 대수는?
sum(np_df>=np_df.mean())

# 평균 mpg 보다 높은(이상) 자동차 모델은?
model_names[np_df>=np_df.mean()]

# 평균 mpg 보다 낮은(미만) 자동차 모델은?
model_names[np_df<np_df.mean()]

# 15 작거나 22이상인 데이터 개수는?
sum((np_df<15) | (np_df>22))

np.random.seed(2024)
a = np.random.randint(1,10000, 5)
b = np.array(["A","B","C","F","W"])

# a[조건을 만족하는 논리형 벡터]
a[(a>2000) & (a<5000)]
b[(a>2000) & (a<5000)]

a[a>3000] = 3000
a

# where
np.random.seed(2024)
a = np.random.randint(1,26346, 1000)
a

# 처음으로 22000보다 큰 숫자가 나왔을 때,
# 숫자 위치와 그 숫자는 무엇인가요?
x = np.where(a>22000)
x
type(x)

my_index = x[0][0] #숫자의 위치
a[my_index] #그 숫자의 값

# 처음으로 10000보다 큰 숫자들 중 나왔을 때,
# 50번째 숫자 위치와 그 숫자는 무엇인가요?
y = np.where(a>10000)
y
index = y[0][49] #숫자의 위치
a[index] #그 숫자의 값

# 500보다 작은 숫자들 중
# 가장 마지막으로 나오는 숫자 위치와 그 숫자는 무엇인가요?
z = np.where(a<500)
z
z_index = z[0][-1]
a[z_index]

a = np.array([20, np.nan, 13, 24, 309])
a + 3
np.nan + 3
np.mean(a)
np.nanmean(a)
# ?np.nan_to_num
np.nan_to_num(a, nan=0)


False
a = None
b = np.nan
b
a

b + 1
a + 1

a = np.array([20, np.nan, 13, 24, 309])
~np.isnan(a)
a_filtered = a[~np.isnan(a)]
a_filtered

str_vec = np.array(["사과", "배", "수박", "참외"])
str_vec
str_vec[[0, 2]]

mix_vec = np.array(["사과", 12, "수박", "참외"], dtype=str)
mix_vec

combined_vec = np.concatenate((str_vec, mix_vec))
combined_vec

combined_vec_list = np.concatenate([str_vec, mix_vec])
combined_vec_list

# dtype='<U2' : U Unicode String

# 벡터들을 세로로 붙여줍니다.
col_stacked = np.column_stack((np.arange(1, 5), 
                                np.arange(12, 16)))
col_stacked

# 벡터들을 가로로 쌓아줍니다.
row_stacked = np.row_stack((np.arange(1, 5), 
                                np.arange(12, 16)))
row_stacked

# <string>:1: DeprecationWarning: `row_stack` alias is deprecated. Use `np.vstack` directly.

# 길이가 다른 벡터 합치기
uneven_stacked = np.column_stack((np.arange(1, 5), # 1,2,3,4
                                  np.arange(12, 18))) # 12,13,14,15,16,17
uneven_stacked

vec1 = np.arange(1,5)
vec2 = np.arange(12, 18)

np.resize(vec1, len(vec2))
vec1 = np.resize(vec1, len(vec2))
vec1

# len이용해 길이 맞추고 다시 세로로 합치기
stacked = np.column_stack((vec1, vec2))
stacked

# len이용해 길이 맞추고 다시 가로로 쌓기
row_stacked = np.row_stack((vec1, vec2))
row_stacked

# 연습문제

# 1 주어진 벡터의 각 요소에 5를 더한 새로운 벡터 생성
a = np.array([1, 2, 3, 4, 5])
a
a + 5 

# 2 주어진 벡터의 홀수 번째 요소 추출
a = np.array([12, 21, 35, 48, 5])
a
a[0::2]

# 3 최대값 찾기
a = np.array([1, 22, 93, 64, 54])
a
a.max()

# 4 중복된 값 제거한 새로운 벡터
a = np.array([1, 2, 3, 2, 4, 5, 4, 6])
np.unique(a)

# 5 주어진 두 벡터의 요소를 번갈아 가면서 합쳐서 새로운 벡터를 생성
a = np.array([21, 31, 58])
b = np.array([24, 44, 67])
a
b

# array([24, 44, 67])
x = np.empty(6)
x

# 짝수
x[1::2] = b

# 홀수
x[0::2] = a

x
