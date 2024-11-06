# 리스트 예제
fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "apple", 3.5, True]

# 빈 리스트 생성
empty_list1 = []
empty_list2 = list()

# 초기값을 가진 리스트 생성
numbers = [1, 2, 3, 4, 5]
range_list = list(range(5))
range_list

range_list[3] = "LS 빅데이터 스쿨"
range_list

# 두번째 원소에 다음과 같은 정보를 넣어보자
# ["1st", "2nd", "3rd"]
range_list[1] = ["1st", "2nd", "3rd"]
range_list

# "3rd" 만 가져오고 싶다면?
range_list[1][2]

# 리스트 내포(Comprehension)
# 1. 대괄호로 쌓여져 있다 => 리스트다.
# 2. 넣고 싶은 수식표현을 x을 사용해서 표현
# 3. for .. in .. 을 사용해서 원소정보 제공

list(range(10))
squares = [x**2 for x in range(10)]
squares

my_squares = [x **3 for x in [3,5,2,15]]
my_squares

import numpy as np
my_squares = [x **3 for x in np.array([3,5,2,15])]
my_squares

# Pandas Series
import pandas as pd

# Pandas 시리즈가 in 값에 들어가도 가능한가?
exam = pd.read_csv("data/exam.csv")
exam["math"] # 데이터 타입 : 시리즈
my_squares = [x **3 for x in exam["math"]]
my_squares

# 리스트 합치기
3 + 2 
"안녕" + "하세요"
"안녕" * 3

# 리스트 연결
list1 = [1,2,3]
list2 = [4,5,6]
combined_list = list1 + list2

(list1 * 3) + (list2 * 5)

numbers = [5,2,3]
# [x for x in numbers]
# >>>[5, 2, 3]
repeated_list = [x for x in numbers for y in [4,2,1,3]]
repeated_list = [x for x in numbers for y in range(4)]
repeated_list

# _의 의미
# 1. 앞에 나온 값을 가리킴
5 + 4
_ + 6 # _는 9를 의미


# 2. 값 생략, 자리 차지(placeholder) 
a, _, b = (1, 2, 4)
a; b

# _
# _ 가 변수가 되버림
# _ = None
# del _

# for loop 문법
# for i in 범위:
#   작동방식
for x in [4,1,2,3]:
    print(x)
    


for i in range(5):
    print(i**2)


# mylist = []
# for i in [1,2,3] : 
#   mylist.append(i*2)

mylist=[]
mylist.append(2)
mylist.append(4)
mylist.append(6)

# 리스트를 하나 만들어서 
# for loop 사용해서 2,4,6,8, ..., 20의 수를
# 채워넣어보세요

# 1.
# [i for i in range(2,21,2)]

# 2.
mylist=[]
for i in range(1,11) :
    mylist.append(i*2)
mylist

# 3.
mylist = [0] * 10
for i in range(10) : 
    mylist[i] = (i + 1) * 2
mylist

# 4. 인덱스 공유해서 카피하기
mylist_b = [2, 4, 6, 80, 10, 12, 24, 35, 23, 20]
mylist = [0] * 10
for i in range(10) :
    mylist[i] = mylist_b[i]
mylist


# 퀴즈: mylist_b의 홀수번째 위치에 있는 숫자들만 mylist에 가져오기
mylist_b = [2, 4, 6, 80, 10, 12, 24, 35, 23, 20]
mylist = [0] * 5

for i in range(5) : 
    mylist[i] = mylist_b[i*2]
mylist

# 리스트 컴프리헨션으로 바꾸는 방법
# 바깥은 무조건 대괄호로 묶어줌 :리스트 반환하기 위해서.
# for 루프의 :는 생략
# 실행 부분을 먼저 써준다.
# 결과값을 발생하는 표현만 남겨두기

mylist = [i*2 for i in range(1,11)]
mylist

[x for x in numbers]

for i in range(5):
    print("hello")
    

# i : 0,1,2
# j : 0,1
# for i in range(3):
#     for j in range(2):
#         print(i, j)

for i in [0,1,2]:
    for j in [0,1]:
        print(i, j)


for i in [0,1]:
    for j in [4,5,6]:
        print(i)
        

for i in numbers:
    for j in range(4):
        print(i, j)
        
for i in numbers:
    for j in range(4):
        print(i)


# 리스트 컴프리헨션 변환        
[i for i in numbers for j in range(4)]


numbers = [5,2,3]
# [x for x in numbers]
# >>>[5, 2, 3]
repeated_list = [x for x in numbers for y in [4,2,1,3]]
repeated_list

# 원소 체크
fruits = ["apple", "banana", "cherry"]
fruits
"banana" in fruits

# [x == "banana" for x in fruits]

mylist=[]
for x in fruits:
    mylist.append(x == "banana")
mylist

# banana의 위치를 뱉어내게 하려면? 
fruits = ["apple", "apple", "banana", "cherry"]

import numpy as np

fruits = np.array(fruits)
my_index = np.where(fruits == "banana")
int(my_index[0][0])

# 원소 거꾸로 써주는 reverse()
fruits = ["apple", "apple", "banana", "cherry"]
fruits.reverse()
fruits

# 원소 맨 끝에 붙여주기
fruits.append("pineapple")
fruits

# 원소 삽임
fruits.insert(2, "test")
fruits

# 원소 제거
fruits.remove("test")
fruits

fruits.remove("apple")
fruits

import numpy as np

# 넘파이 배열 생성
fruits = np.array(["apple", "banana", "cherry", "apple", "pineapple"])

# 제거할 항목 리스트
items_to_remove = np.array(["banana", "apple"])

# 마스크(논리형 벡터) 생성
mask = ~np.isin(fruits, items_to_remove)
mask = ~np.isin(fruits, ["banana", "apple"])
# np.isin(fruits, items_to_remove)

# 불리언 마스크를 사용하여 항목 제거
filtered_fruits = fruits[mask]
print("remove() 후 배열:", filtered_fruits)
