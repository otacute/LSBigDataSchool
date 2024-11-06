# 파이썬 분석 교재
import pandas as pd
import numpy as np

# 데이터 탐색 함수
# head()
# tail()
# shape
# info()
# describe()

exam = pd.read_csv('data/exam.csv')
exam.head()
exam.tail()

# 메서드 vs. 속성(어트리뷰트)
exam.shape
exam.info()
exam.describe()

type(exam)
var = [1,2,3]
type(var)
exam.head()
# var.head()

exam2 = exam.copy()
exam2 = exam2.rename(columns = {'nclass' : 'class'})
exam2

exam2['total'] = exam2['math'] + exam2['english'] + exam2['science']
exam2.head()

exam2["test2"] = np.where(exam2['total'] >= 200, "A", 
                          np.where(exam2['total']>=100, "B", "C"))
# 200 이상 : A
# 100 이상 : B
# 100 미만 : C
exam2.head()

import matplotlib.pyplot as plt
exam2["test2"].value_counts().plot.bar(rot=0)

# help
?exam2.value_counts().plot.bar

plt.show()
plt.clf()

exam2["test2"].isin(["A", "C"])

# 2024.07.16 오늘의 궁금증
# random.randint 발생 시 중복 없이 하려면?
import numpy as np

# 벡터 슬라이싱 예제, a를 랜덤하게 채움
np.random.seed(2024)
a = np.random.randint(1,21,10)
a
?np.random.randint

# 오늘의 질문 : random으로 중복없이 값을 추출하는 방법
# 구글 검색 : np.random.randint seed without replacement
# numpy.random.choice
# random.choice(a, size=None, replace=True, p=None)
# https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html

b = np.random.choice(np.arange(1,21), 10, replace=False)
print(b)

c = np.random.choice(np.arange(1,4), 100, True, np.array([2/5, 2/5, 1/5]))
print(c)

sum(c == 3)
sum(c == 2)
sum(c == 1)
# 뽑힌 수의 비율 비교 가능

# 데이터 전처리 함수
# query()
# df[]
# sort_values()
# groupby()
# assign()
# agg()
# merge()
# concat()

exam = pd.read_csv("data/exam.csv")

# 조건에 맞는 행을 걸러내는 .query()
# exam[exam["nclass"]==1]
exam.query("nclass == 1")
exam.query("nclass == 2")
exam.query("nclass != 1")
exam.query("nclass != 3")

exam.query("math > 50")
exam.query("math < 50")
exam.query("english >= 50")
exam.query("english <= 80")

exam.query('nclass == 1 & math >=50')
exam.query('nclass == 2 & english >=80')

exam.query('math >=90 | english >=90')
exam.query('english < 90 | science < 50')

exam.query('nclass == 1 | nclass == 3 | nclass ==5') 
exam.query('nclass in [1,3,5]')
exam.query('nclass not in [1,2]')
# exam[~exam['nclass'].isin([1,2])]

exam["nclass"]
type(exam["nclass"])
exam[["nclass"]]
type(exam[["nclass"]])

exam[["id","nclass"]]
exam.drop(columns = "math")
exam

exam.query("nclass == 1")[["math","english"]]

exam.query("nclass == 1")\
     [["math","english"]]\
     .head()

# 정렬하기
exam.sort_values("math")
exam.sort_values("math", ascending=False)
exam.sort_values(["nclass","english"], ascending=[True, False])

# 변수추가
exam = exam.assign(
    total = exam["math"] + exam["english"] + exam["science"],
    mean = (exam["math"] + exam["english"] + exam["science"])/3)\
    .sort_values("total", ascending=False)
exam

# lambda 함수 사용하기
exam2 = pd.read_csv("data/exam.csv")
exam2 = exam2.assign(
    total = lambda x: x["math"] + x["english"] + x["science"],
    mean = lambda x: (x["math"] + x["english"] + x["science"])/3)\
    .sort_values("total", ascending=False)
exam

# 요약을 하는 .agg()
exam2.agg(mean_math = ("math","mean"))
exam2.groupby("nclass")\
     .agg(mean_math=("math","mean"))
     
# 반 별 각 과목 별 평균
exam2.groupby("nclass")\
     .agg(mean_math=("math","mean"),
     mean_eng=("english","mean"),
     mean_sci=("science","mean")
     )

! pip install pydataset
import pydataset
df = pydataset.data("mpg")
df

df.groupby('drv')\
  .agg(n=('drv','count'))

df['drv'].value_counts()

type(df['drv'].value_counts())

df['drv'].value_counts().query('n>100')
# AttributeError: 'Series' object has no attribute 'query'

df['drv'].value_counts()\
         .to_frame('n')\
         .query('n>100')


# 숙제 : p144, p153, p158
