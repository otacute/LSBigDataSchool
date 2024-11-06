import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


route = pd.read_excel("C:/Users/USER/Documents/LS빅데이터스쿨/team3_project/youna/youna_route.xlsx",header=1)
route.head(10)

# 변수 변경
route = route.rename(columns = {"시점":"year", "항목":"access_path", "소계" : "total", "남자" : "male", "여자" : "female"})

route.head()

# 연도 결측치 처리
route.loc[0:8,"year"] = 2018
route.loc[9:17,"year"] = 2019
route.loc[18:26,"year"] = 2020
route.loc[27:35,"year"] = 2021
route.loc[36:43,"year"] = 2022
route["year"]=route["year"].astype(int)
route

# 항목
route.loc[route["access_path"]=="과거 방문 경험", "access_path"] = "experience"
route.loc[route["access_path"]=="주변인(친지/친구/동료 등)", "access_path"] = "acquaintance"
route.loc[route["access_path"]=="인터넷 사이트/모바일 앱(PC/스마트폰)", "access_path"] = "internet_mobile_app"
route.loc[route["access_path"]=="광고(TV/라디오/ 신문/잡지/ 지하철/옥외 광고판 등)", "access_path"] = "advertising"
route.loc[route["access_path"]=="기사 및 방송 프로그램(TV/라디오/ 신문/잡지)", "access_path"] = "article_broadcast"
route.loc[route["access_path"]=="관광 안내 서적", "access_path"] = "guidebook"
route.loc[route["access_path"]=="여행사(방문, 전화)", "access_path"] = "travel_agency"
route.loc[route["access_path"]=="정보 없이 방문", "access_path"] = "no_information"
route.loc[route["access_path"]=="기타", "access_path"] = "etc"
route.head()

# 결측치 제거
route.replace('-', np.nan, inplace=True)
route.isna().sum()

# 연령대
#route = route.rename(columns = {"15~19세":"teens", "20대":"20years", "30대" : "30years", 
#                                "40대":"40years", "50대":"50years", "60대":"60years",
#                               "70세 이상":"70years+"})
  
route["teens"] = route["15~19세"]   
route["young_adults"] = route["20대"] + route["30대"]
route["middle_adults"] = route["40대"] + route["50대"]
route["seniors"] = route["60대"] + route["70세 이상"]
route = route.drop(['15~19세', '20대', '30대', '40대', '50대', '60대', '70세 이상'], axis=1)
route.info()


# 월급
route["l_sal"] = route["100만원 미만"] + route["100~200만원 미만"]
route["m_sal"] = route["200~300만원 미만"] + route["300~400만원 미만"] + route["400~500만원 미만"]
route["h_sal"] = route["500~600만원 미만"] + route["600만원 이상"]
route["nr"]= route["무응답"]

route = route.drop(['100만원 미만', '100~200만원 미만', '200~300만원 미만', '300~400만원 미만', '400~500만원 미만',\
                '500~600만원 미만', '600만원 이상', "무응답"], axis=1)
                

# 가구
route = route.rename(columns = { '1인' : 'per1',
						 '2인' : 'per2',
						'3인이상' : 'per3+'})

# 학벌
route = route.rename(columns = { '초졸 이하' : 'elmt',
						 '중학교' : 'mid',
						'고등학교' : 'high',
						'대학교이상' : 'univ+',
						})
route.info()

# 직업
route = route.drop(['임금봉급근로자', '고용원있는사업주', '고용원없는자영업자', 
                    '무급가족 종사자', '사무전문', '기술생산노무', '판매서비스',
                    "자영업","전업주부","학생","무직은퇴","기타"], axis=1)
route

# 결측치 제거
route.replace('-', np.nan, inplace=True)
route.isna().sum()

# 전처리한 데이터 파일 추출
route.to_excel('pre_route.xlsx', index=False)



# 경로 평균 백분율 막대그래프
## 1. 5개년 평균 경로 순위 구하기
# route2["access_path"] = route["access_path"].drop_duplicates()
# route2= route.groupby("access_path").agg(total_mean=("total","mean")).reset_index()
# route2

#route2 = pd.DataFrame()
route2=route.groupby("access_path")\
.agg(total_mean = ("total","mean"))
route2


route2 = route2.sort_values(["total_mean"], ascending = False)
route2

## 2. 그래프화 화기


plt.clf()
plt.subplots_adjust(bottom=0.2)
sns.barplot(data=route2, x="access_path", y="total_mean", palette = "Set1")
plt.xlabel('여행 정보 획득 경로')
plt.ylabel('소계')
plt.xticks(rotation=20)
plt.xticks(fontsize=8)
plt.rcParams['font.family'] ='Malgun Gothic'
plt.title('5개년 평균 여행 정보 획득 경로',fontsize=15)
plt.show()


# 연도별로 정보 획득 경로 순위 어떻게 변하는지
route_path = route.groupby(['year', 'access_path'])['total'].sum().unstack()
plt.clf()

plt.figure(figsize=(12, 6))
route_path.plot(ylim=[7,45])

plt.subplots_adjust(right=0.71)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1),fontsize=7)
plt.rcParams['font.family'] ='Malgun Gothic'
plt.title('연도 별 여행 정보 획득 경로',fontsize=15)
plt.show()



# 경로 Top 4에서는 어떤 가구수가 많이 이용하는지 -> 연도합평균
route_p = route[route['access_path'].isin(['acquaintance', 'experience', 'no_information', 'internet_mobile_app'])]
route_per = route_p.groupby('access_path')[["per1","per2","per3+"]].mean()
route_per

path = route_p['access_path'].unique()
index = np.arange(len(path))
bar_width=0.2

plt.clf()
plt.bar(index, route_per["per1"], bar_width, label="per1")
plt.bar(index + bar_width, route_per["per2"], bar_width, label="per2")
plt.bar(index + 2 * bar_width, route_per["per3+"], bar_width, label="per3+")

plt.rcParams['font.family'] ='Malgun Gothic'
plt.xlabel('여행 정보 획득 경로 Top 4')
plt.title('여행 정보 획득 경로 Top 4 : 가구원 수',fontsize=15)
plt.xticks(index + bar_width, path)
plt.legend()
plt.show()




route_per=route.groupby("access_path")\.agg(total_mean = ("total","mean"))
route_per




# 경로 Top 4에서는 어떤 성별이 많이 이용하는지 -> 연도합평균





# 경로 Top 4에서는 어떤 연령대가 많이 이용하는지 -> 연도합평균







# 경로 Top 3에서는 어떤 소득층이 많이 이용하는지 -> 연도합평균
## 1등-지인추천
route_exp = route[route['access_path'] == 'acquaintance']
route_exp = route_exp.groupby('year')[['l_sal', 'm_sal', 'h_sal']].mean()

years = route_exp['year'].unique()
index = np.arange(len(years))
bar_width=0.2

## 막대 그래프 그리기
plt.clf()

plt.bar(index, route_exp['l_sal'], bar_width, label='l_sal')
plt.bar(index + bar_width, route_exp['m_sal'], bar_width, label='m_sal')
plt.bar(index + 2 * bar_width, route_exp['h_sal'], bar_width, label='h_sal')

plt.xlabel('Year')
plt.ylabel('Salary')
plt.title('Yearly Travel Information Acquisition Rate by Income:Acquaintance',fontsize=10)
plt.xticks(index + bar_width, years)
plt.legend()
plt.show()


## 2등-경험
route_exp = route[route['access_path'] == 'experience']
route_exp = route_exp.groupby('year')[['l_sal', 'm_sal', 'h_sal']].mean()

years = route_exp['year'].unique()
index = np.arange(len(years))
bar_width=0.2

## 막대 그래프 그리기
plt.clf()

plt.bar(index, route_exp['l_sal'], bar_width, label='l_sal')
plt.bar(index + bar_width, route_exp['m_sal'], bar_width, label='m_sal')
plt.bar(index + 2 * bar_width, route_exp['h_sal'], bar_width, label='h_sal')

plt.xlabel('Year')
plt.ylabel('Salary')
plt.title('Yearly Travel Information Acquisition Rate by Income:Experience',fontsize=10)
plt.xticks(index + bar_width, years)
plt.legend()
plt.show()


## 3등-정보없이
route_exp = route[route['access_path'] == 'no_information']
route_exp = route_exp.groupby('year')[['l_sal', 'm_sal', 'h_sal']].mean()

years = route_exp['year'].unique()
index = np.arange(len(years))
bar_width=0.2

## 막대 그래프 그리기
plt.clf()

plt.bar(index, route_exp['l_sal'], bar_width, label='l_sal')
plt.bar(index + bar_width, route_exp['m_sal'], bar_width, label='m_sal')
plt.bar(index + 2 * bar_width, route_exp['h_sal'], bar_width, label='h_sal')

plt.xlabel('Year')
plt.ylabel('Salary')
plt.title('Yearly Travel Information Acquisition Rate by Income:No_information',fontsize=10)
plt.xticks(index + bar_width, years)
plt.legend()
plt.show()


# 경로 순위
route.sort_values(["total"],ascending=False).head(5)
route.head(10)
