# p168
# 중간고사 데이터 만들기
import pandas as pd
test1 = pd.DataFrame({ 'id' : [1,2,3,4,5],
                       'midterm' : [60,80,70,90,85]})
                       
# 기말고사 데이터 만들기
test2 = pd.DataFrame({ 'id' : [1,2,3,40,5],
                       'final' : [70,83,65,95,80]})
                       
test1
test2

# Left Join **(중요)**
total = pd.merge(test1, test2, how="left", on="id")
total

# Right Join
total = pd.merge(test1, test2, how="right", on="id")
total

# Inner Join
total = pd.merge(test1, test2, how="inner", on="id")
total

# Outer Join
total = pd.merge(test1, test2, how="outer", on="id")
total

# 다른 데이터를 이용해 변수 추가하기
exam = pd.read_csv('data/exam.csv')
exam

name = pd.DataFrame({'nclass' : [1,2,3,4,5],
                     'teacher' :['kim','lee','park','choi','jung']})
name

exam = pd.merge(exam, name, how="left", on="nclass")
exam

# 데이터를 세로로 쌓는 방법
score1 = pd.DataFrame({ 'id' : [1,2,3,4,5],
                       'midterm' : [60,80,70,90,85]})
                       
score2 = pd.DataFrame({ 'id' : [6,7,8,9,10],
                       'final' : [70,83,65,95,80]})

score1
score2
score_all = pd.concat([score1, score2])
score_all

# concat 옆으로 붙이기
test1
test2
pd.concat([test1, test2], axis=1)

import pandas as pd
import numpy as np

df = pd.DataFrame({"sex"   :["M","F", np.nan, "M","F"],
                   "score" : [5,4,3,4,np.nan]})
df

# NaN인 위치에만 True를 반환
pd.isna(df)

df["score"] + 1

pd.isna(df).sum()
pd.isna(df).sum().sum()

# 결측치 제거하기
df.dropna()                          # 모든 변수 결측치 제거
df.dropna(subset = "score")          # score변수에서 결측치 제거
df.dropna(subset = ["score", "sex"]) # 여러 변수 결측치 제거법

exam = pd.read_csv("data/exam.csv")
exam

# 데이터 프레임 location을 사용한 인덱싱
# exam.loc[행 인덱스, 열 인덱스](리스트)
# exam.iloc[행 인덱스, 열 인덱스] (숫자)
exam.loc[0,0]
exam.loc[[0], ["id","nclass"]]
exam.iloc[0:2, 0:4]

exam.loc[[2,7,4], ['math']] = np.nan
exam.iloc[[2,7,4],2] = 3
exam

# score의 value가 3.0인데 4.0으로 바꾸고 싶을 때
df.loc[df["score"] == 3.0, ["score"]] = 4
df

# 수학점수가 50점 이하인 학생들 점수를 50점으로 상향 조정!
exam
exam.loc[exam['math'] <=50, ['math']] = 50
exam

# 영어 점수가 90점 이상인 학생들 90으로 하향 조절
# iloc 조회는 안됨
exam.loc[exam['english']>=90, "english"]

#iloc을 사용해서 조회하려면 무조건 숫자 벡터가 들어가야 함
exam.iloc[exam['english']>=90, 3] = 90              # 실행 안됨
exam.iloc[np.array(exam['english']>=90), 3]         # 실행 됨
exam.iloc[np.where(exam['english']>=90)[0], 3]      # np.where 도 튜플이라 [0] 꺼내오면 됨
exam.iloc[exam[exam['english'] >= 90].index, 3]     # index 벡터도 작동

exam = pd.read_csv("data/exam.csv")

# math 점수 50점 이하 - 로 변경
exam.loc[exam['math']<=50, ['math']] = "-"
exam

# "-" 결측치를 수학점수 평균으로 바꾸고 싶은 경우
#1 
math_mean = exam.loc[exam['math']!="-", 'math'].mean()
exam.loc[exam['math'] == '-', 'math'] = math_mean
exam
 
#2
math_mean = exam.loc[exam['math']=='-', 'math'] = exam.query('math not in ["-"]')['math'].mean()
exam.loc[exam['math'] == '-', 'math'] = math_mean
exam

#3
math_mean = exam[exam['math']!="-"]["math"].mean()
exam.loc[exam['math']=='-', 'math'] = math_mean
exam

#4
exam.loc[exam['math']=="-", ['math']] = np.nan
math_mean = exam["math"].mean()
exam.loc[pd.isna(exam['math']), ['math']] = math_mean
exam

#5
math_mean = np.nanmean(np.array([np.nan if x == '-' else float(x) for x in exam["math"]]))
#vector = np.array([float(x) if x!="-" else np.nan for x in exam["math"]])
exam['math'] = np.where(exam['math'] == "-", math_mean, exam["math"])
exam

#6
math_mean = exam[exam["math"]!= "-"]["math"].mean()
exam["math"] = exam["math"].replace("-", math_mean)
exam
