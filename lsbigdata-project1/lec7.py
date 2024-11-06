# 쉽게 배우는 파이썬 데이터 분석 77p

import pandas as pd
import numpy as np

df = pd.DataFrame({'name' :['김지홍', '이유진', '박동현', '김민지'],
             'english' : [90, 80, 60, 70],
             'math' : [50,60,100,20]})
df

type(df)
type(df["name"])

sum(df["english"])

df2 = pd.DataFrame({'제품' : ['사과','딸기','수박'],
                    '가격' :[1800, 1500, 3000],
                    '판매량' :[24, 38, 13]})
df2

df2['가격'].mean()
df2['판매량'].mean()

pd.show_versions()

import pandas as pd
! pip install openpyxl

df_exam = pd.read_excel("data/excel_exam.xlsx")
df_exam

sum(df_exam["math"])   /20
sum(df_exam["english"])/20
sum(df_exam["science"])/20

len(df_exam)
df_exam.shape
df_exam.size # 전체 원소 갯수

?pd.read_excel

# sheet2를 읽는 방법
# df_exam = pd.read_excel("data/excel_exam.xlsx", sheet_name="Sheet2")
# df_exam

df_exam["total"] = df_exam["math"] + df_exam["english"] + df_exam["science"]

df_exam["mean"] = df_exam["total"] / 3

df_exam[df_exam["math"] > 50]

df_exam[(df_exam["math"] > 50) & (df_exam["english"] > 50)]

# 수학을 잘하는 애들(평균보다 높은 애) 중에서 영어는 평균보다 못하는 애가 몇명일까?
df_exam["math"].mean() # 수학 평균 57.45
df_exam["english"].mean() # 영어 평균 - 84.9
df_exam[(df_exam["math"] > df_exam["math"].mean()) &
        (df_exam["english"] < df_exam["english"].mean())]

df_nc3 = df_exam[df_exam['nclass'] == 3]
df_nc3[["math", "english", "science"]]
df_nc3[1:4]
df_nc3[1:2]
df_nc3[0:1]

import numpy as np
a = np.array([4,2,5,3,6])
a[2]

df_exam[0:10]
df_exam[7:16]

df_exam[0:10:2] # 홀수행
df_exam[7:16]

df_exam.sort_values("math", ascending=False)
df_exam.sort_values(["nclass","math"], ascending=[True, False])

np.where(a > 3, "Up", "Down")
df_exam["updown"] = np.where(df_exam["math"] > 50, "Up", "Down")
df_exam

