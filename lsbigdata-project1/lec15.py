# 숙제 확인 코드

# google sheet pandas로 읽어오는 방법

import pandas as pd

# https://docs.google.com/spreadsheets/d/1RC8K0nzfpR3anLXpgtb8VDjEXtZ922N5N0LcSY5KMx8/edit?gid=158309216#gid=158309216
gsheet_url = "https://docs.google.com/spreadsheets/d/1RC8K0nzfpR3anLXpgtb8VDjEXtZ922N5N0LcSY5KMx8/gviz/tq?tqx=out:csv&sheet=Sheet2"

df = pd.read_csv(gsheet_url)
df.head()

# 랜덤하게 2명을 뽑아서 보여주는 코드
import numpy as np
np.random.seed(20240730)
np.random.choice(df["이름"], 2, replace=False)

# 교재 225p
!pip install pyreadstat
import pandas as pd
import numpy as np
import seaborn as sns

raw_welfare = pd.read_spss("./data/koweps/Koweps_hpwc14_2019_beta2.sav")
welfare = raw_welfare.copy()
welfare.shape
# welfare.describe()

welfare = welfare.rename(
    columns = {
        "h14_g3" : "sex", 
        "h14_g4" : "birth",
        "h14_g10" : "marriage_type",
        "h14_g11" : "religion",
        "p1402_8aq1" : "income",
        "h14_eco9" : "code_job",
        "h14_reg7" : "code_region"
    }
)

welfare = welfare[["sex", "birth", "marriage_type",
                    "religion","income","code_job","code_region"]]
                    
welfare['sex']
welfare['sex'].dtypes
welfare['sex'].value_counts()
# welfare['sex'].isna().sum()

welfare['sex'] = np.where(welfare['sex'] == 1, 'male', 'female')
welfare

welfare['income'].describe()
welfare['income'].isna().sum()
welfare['income'].value_counts()

sex_income = welfare.dropna(subset="income")\
       .groupby("sex", as_index=False)\
       .agg(mean_income = ("income","mean"))

sex_income

import seaborn as sns
sns.barplot(data=sex_income, x="sex", y="mean_income", hue="sex")
plt.show()
plt.clf()

# 숙제5 : chapter 9-2 설문조사
# 위 그래프에서 각 성별 95% 신뢰구간 계산 후 그리기
# 모분산은 표본 분산을 사용해서 추정
# norm.ppf() 사용해서 그릴 것
# 위, 아래 검정색 막대기로 표시

welfare["birth"].describe()
sns.histplot(data=welfare, x="birth")
plt.show()
plt.clf()

welfare["birth"].isna().sum()

welfare = welfare.assign(age = 2019 - welfare["birth"] + 1)
welfare["age"]
sns.histplot(data=welfare, x="age")
plt.show()
plt.clf()

import seaborn as sns
age_income = welfare.dropna(subset = "income")\
                    .groupby("age")\
                    .agg(mean_income = ("income","mean"))
age_income.head()

sns.lineplot(data=age_income, x='age', y='mean_income')
plt.show()
plt.clf()

# 나이별 income 칼럼에서 na 개수 세어보기
# 나이별 무 응답자 갯수
my_df = welfare.assign(income_na = welfare["income"].isna())\
                            .groupby("age", as_index=False)\
                            .agg(n=("income_na","sum"))

sns.barplot(data=my_df, x="age", y="n")
plt.show()
plt.clf()

# 교재 240p - 연령대에 따른 월급 차이
# 나이 변수 살펴보기
welfare['age'].head()

# 연령대 변수 만들기
welfare = welfare.assign(ageg = np.where(welfare['age'] < 30, 'young',
                                np.where(welfare['age'] <=59, 'middle',
                                                               'old')))

# 빈도 구하기
welfare['ageg'].value_counts()

# 빈도 막대 그래프 만들기
sns.countplot(data = welfare, x = 'ageg', hue='ageg')
plt.show()
plt.clf()

# 연령대별 월급 평균표 만들기
ageg_income = welfare.dropna(subset = 'income')\
                     .groupby('ageg', as_index=False)\
                     .agg(mean_income =('income','mean'))
ageg_income                     
                     
# 막대 그래프 만들기
sns.barplot(data = ageg_income, x = 'ageg', y='mean_income', hue='ageg')
plt.show()
plt.clf()

# 막대 정렬하기
sns.barplot(data = ageg_income, x ='ageg', y='mean_income',
            order = ['young', 'middle','old'], hue='ageg')
plt.show()
plt.clf()

# 나이대별 수입 분석
# cut
# ?pd.cut
bin_cut = np.array([0,9,19,29,39,49,59,69,79,89,99,109,119])

welfare = welfare.assign(age_group = pd.cut(welfare["age"], 
               bins=bin_cut,
               labels=(np.arange(12) * 10).astype(str) + "대"))

age_income = welfare.dropna(subset = "income")\
                    .groupby("age_group", as_index=False)\
                    .agg(mean_income = ("income","mean"))
age_income
sns.barplot(data=age_income, x="age_group", y="mean_income")
plt.show()
plt.clf()

# 내가 짠 코드 - max값을 써서 연령 카테고리 설정하는 방법
age_min, age_max = (welfare['age'].min(), welfare['age'].max())
age_max // 10

bin_cut = [0] + [10 * i + 9 for i in np.arange(age_max//10 + 1)]

vec_x = np.random.randint(0, 100, 50)
pd.cut(vec_x, bins=bin_cut)

# 244p 연령대 및 성별 월급 차이 분석하기
# welfare["age_group"]의 dtype: category
welfare["age_group"]

welfare["age_group"] = welfare["age_group"].astype("object")
# 판다스 데이터 프레임을 다룰 때, 변수의 타입이
# 카테고리로 설정되어 있는 경우, groupby + agg 콤보가 안먹힘
# 그래서 object 타입으로 바꿔 준 후 수행

welfare["age_group"] = welfare["age_group"].astype("object")
sex_age_income = \
    welfare.dropna(subset="income")\
    .groupby(["age_group","sex"], as_index=False)\
    .agg(mean_income=("income","mean"))

sex_age_income

sns.barplot(data=sex_age_income,
            x="age_group", y="mean_income",
            hue="sex")
plt.show()
plt.clf()

# 연령대별, 성별 상위 4% 수입 찾아보세요!
# quantile 함수 설명
x = np.arange(10)
np.quantile(x, q=0.7)
np.quantile(x, q=0.5)

welfare["age_group"] = welfare["age_group"].astype("object")

# lambda x: 녀석을 어떻게 쓰는지에 대한 설명 예제 코드
def my_f(vec):
    return vec.sum()

sex_age_income = \
    welfare.dropna(subset="income") \
    .groupby(["age_group", "sex"], as_index=False) \
    .agg(top4per_income=("income", lambda x: my_f(x)))
sex_age_income

# 진짜 연령대별, 성별 상위 4% 수입 찾아보세요
sex_age_income = \
    welfare.dropna(subset="income")\
    .groupby(["age_group","sex"], as_index=False)\
    .agg(top4per_income=("income",
                          lambda x : np.quantile(x, q=0.96)))

sex_age_income

sns.barplot(data=sex_age_income,
            x="age_group", y="top4per_income",
            hue="sex")
plt.show()
plt.clf()

# 2024 07 31(수) 오후 수업 #

## 참고
welfare.dropna(subset='income')\
       .groupby('sex', as_index=False)['income']\
       .agg(['mean','std'])

welfare.dropna(subset='income')\
       .groupby('sex', as_index=False)[['income']]\
       .agg(['mean','std'])
       
welfare.dropna(subset='income')\
       .groupby('sex', as_index=False)['income']\
       .mean()

# 일준님의 안경에서 나온 똑똑 아이디어
my_f(welfare.dropna(subset='income').groupby('sex', as_index=False)['income'])

# 참고 끝

# 9-6장

welfare["code_job"]
welfare["code_job"].value_counts()

list_job = pd.read_excel("./data/koweps/Koweps_Codebook_2019.xlsx",
                            sheet_name="직종코드")
list_job.head()

welfare = welfare.merge(list_job, how="left", on = "code_job")

welfare.dropna(subset=["job","income"])[["income","job"]]

# 직업별 월급 표 만들기 p250
job_income = welfare.dropna(subset = ['job','income'])\
                    .groupby('job', as_index=False)\
                    .agg(mean_income = ('income','mean'))
job_income.head()

top10 = job_income.sort_values('mean_income', ascending=False).head(10)
top10

# 맑은 고딕 폰트 설정
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family' : 'Malgun Gothic'})

sns.barplot(data = top10, y='job', x='mean_income', hue='job')
plt.tight_layout()
plt.show()
plt.clf()

# issac 센세 코드
#1.
df = welfare.dropna(subset = ['job','income'])\
                    .query("sex == 'male'")\
                    .groupby('job', as_index=False)\
                    .agg(mean_income = ('income','mean'))\
                    .sort_values('mean_income', ascending=False)\
                    .head(10)
df

import matplotlib.pyplot as plt
plt.rcParams.update({'font.family' : 'Malgun Gothic'})
sns.barplot(df, y='job', x="mean_income", hue='job')
plt.show()
plt.clf()

#2.
df = welfare.dropna(subset = ['job','income'])\
                    .query("sex == 'female'")\
                    .groupby('job', as_index=False)\
                    .agg(mean_income = ('income','mean'))\
                    .sort_values('mean_income', ascending=False)\
                    .head(10)
df

import matplotlib.pyplot as plt
plt.rcParams.update({'font.family' : 'Malgun Gothic'})
sns.barplot(df, y='job', x="mean_income", hue='job')
plt.show()
plt.clf()

# 262p 종교 유무에 따른 이혼율 분석하기
welfare.info()
welfare["marriage_type"]

df = welfare.query("marriage_type != 5")\
            .groupby('religion', as_index=False)\
            ["marriage_type"]\
            .value_counts()
df

#핵심!
df = welfare.query("marriage_type != 5")\
            .groupby('religion', as_index=False)\
            ["marriage_type"]\
            .value_counts(normalize=True)
df

df = welfare.query("marriage_type == 1")\
            .assign(proportion = df["proportion"]*100)\
            .round(1)
df
