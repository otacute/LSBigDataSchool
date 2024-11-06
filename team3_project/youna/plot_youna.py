import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


route = pd.read_excel("C:/Users/USER/Documents/LS빅데이터스쿨/team3_project/pre_data/pre_route.xlsx")
route.head(10)


# 경로 평균 백분율 막대그래프
## 1. 5개년 평균 경로 순위 구하기
route2=route.groupby("access_path")\
.agg(total_mean = ("total","mean"))
route2

route2 = route2.sort_values(["total_mean"], ascending = False)
route2

## 2. 그래프화 화기
plt.clf()
plt.rcParams['font.family'] ='Malgun Gothic'
plt.subplots_adjust(bottom=0.2)
sns.barplot(data=route2, x="access_path", y="total_mean", palette = "Set1")
plt.xlabel('여행 정보 획득 경로')
plt.ylabel('소계')
plt.xticks(rotation=20)
plt.xticks(fontsize=7)
plt.title('5개년 평균 여행 정보 획득 경로',fontsize=15)
plt.show()




# 연도별로 정보 획득 경로 순위 어떻게 변하는지

pre_route = pd.read_excel("C:/Users/USER/Documents/LS빅데이터스쿨/team3_project/youna/pre_route.xlsx")

route_path1 = pre_route.query("access_path=='acquaintance'")
route_path2 = pre_route.query("access_path=='advertising'")
route_path3 = pre_route.query("access_path=='no_information'")
route_path4 = pre_route.query("access_path=='internet_mobile_app'")

plt.clf()
plt.plot(route_path1["year"], route_path1["total"], marker='o', label="지인 추천")
plt.plot(route_path2["year"], route_path2["total"], marker='o', label="광고(TV, 라디오 등)")
plt.plot(route_path3["year"], route_path3["total"], marker='o', label="정보 없이 여행")
plt.plot(route_path4["year"], route_path4["total"], marker='o', label="인터넷 및 모바일앱")

plt.subplots_adjust(right=0.71)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1),fontsize=7)
plt.rcParams['font.family'] ='Malgun Gothic'
plt.title('연도 별 여행 정보 획득 경로',fontsize=15)
plt.show()









pre_route = pd.read_excel("C:/Users/USER/Documents/LS빅데이터스쿨/team3_project/youna/pre_route.xlsx")

route_path1 = pre_route.query("access_path=='acquaintance'")
route_path2 = pre_route.query("access_path=='advertising'")
route_path3 = pre_route.query("access_path=='no_information'")
route_path4 = pre_route.query("access_path=='internet_mobile_app'")

plt.clf()
plt.plot(route_path1["year"], route_path1["total"], marker='o', label="지인 추천")
plt.plot(route_path2["year"], route_path2["total"], marker='o', label="광고(TV, 라디오 등)")
plt.plot(route_path3["year"], route_path3["total"], marker='o', label="정보 없이 여행")
plt.plot(route_path4["year"], route_path4["total"], marker='o', label="인터넷 및 모바일앱")

plt.subplots_adjust(right=0.71)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1),fontsize=7)
plt.rcParams['font.family'] ='Malgun Gothic'
plt.title('연도 별 여행 정보 획득 경로',fontsize=15)
plt.show()

# 경로 Top 4에서는 가구원 수가 많이 이용하는지 -> 연도합평균
route_p = route[route['access_path'].isin(['acquaintance', 'experience', 'no_information', 'internet_mobile_app'])]
route_per = route_p.groupby('access_path')[['per1', 'per2', 'per3+']].mean()
route_per

path = route_p['access_path'].unique()
index = np.arange(len(path))
bar_width=0.2

plt.clf()
plt.bar(index, route_per["per1"], bar_width, label="per1")
plt.bar(index + bar_width, route_per["per2"], bar_width, label="per2")
plt.bar(index + 2*bar_width, route_per["per3+"], bar_width, label="per3+")

plt.rcParams['font.family'] ='Malgun Gothic'
plt.xlabel('여행 정보 획득 경로')
plt.title('여행 정보 획득 경로 Top 4 : 가구원 수',fontsize=13)
plt.xticks(index + bar_width, path, fontsize=6.5)
plt.legend(fontsize=8)
plt.show()





# 경로 Top 4에서는 어떤 성별이 많이 이용하는지 -> 연도합평균
route_p = route[route['access_path'].isin(['acquaintance', 'experience', 'no_information', 'internet_mobile_app'])]
route_s = route_p.groupby('access_path')[['male', 'female']].mean()
route_s

path = route_p['access_path'].unique()
index = np.arange(len(path))
bar_width=0.2

plt.clf()
plt.bar(index, route_s["male"], bar_width, label="male")
plt.bar(index + bar_width, route_s["female"], bar_width, label="female")

plt.rcParams['font.family'] ='Malgun Gothic'
plt.xlabel('여행 정보 획득 경로')
plt.title('여행 정보 획득 경로 Top 4 : 성별',fontsize=13)
plt.xticks(index + bar_width, path, fontsize=6.5)
plt.legend(fontsize=8)
plt.show()




# 경로 Top 4에서는 어떤 연령대가 많이 이용하는지 -> 연도합평균

route_p = route[route['access_path'].isin(['acquaintance', 'experience', 'no_information', 'internet_mobile_app'])]
route_m = route_p.groupby('access_path')[["teens","young_adults","middle_adults","senior"]].mean()
route_m

path = route_p['access_path'].unique()
index = np.arange(len(path))
bar_width=0.15

plt.clf()
plt.bar(index, route_m["teens"], bar_width, label="teens")
plt.bar(index + bar_width, route_m["young_adults"], bar_width, label="young_adults")
plt.bar(index + 2 * bar_width, route_m["middle_adults"], bar_width, label="middle_adults")
plt.bar(index + 3 * bar_width, route_m["senior"], bar_width, label="senior")

plt.rcParams['font.family'] ='Malgun Gothic'
plt.xlabel('여행 정보 획득 경로')
plt.title('여행 정보 획득 경로 Top 4 : 연령대',fontsize=13)
plt.xticks(index + bar_width, path, fontsize=6)
plt.legend(fontsize=8)
plt.show()


# 경로 Top 4에서는 어떤 소득층이 많이 이용하는지 -> 연도합평균
route_p = route[route['access_path'].isin(['acquaintance', 'experience', 'no_information', 'internet_mobile_app'])]
route_sal = route_p.groupby('access_path')[["l_sal","m_sal","h_sal"]].mean()
route_sal

path = route_p['access_path'].unique()
index = np.arange(len(path))
bar_width=0.2

plt.clf()
plt.bar(index, route_sal["l_sal"], bar_width, label="l_sal")
plt.bar(index + bar_width, route_sal["m_sal"], bar_width, label="m_sal")
plt.bar(index + 2 * bar_width, route_sal["h_sal"], bar_width, label="h_sal")

plt.rcParams['font.family'] ='Malgun Gothic'
plt.xlabel('여행 정보 획득 경로')
plt.title('여행 정보 획득 경로 Top 4 : 소득층',fontsize=13)
plt.xticks(index + bar_width, path, fontsize=6)
plt.legend(fontsize=8)
plt.show()


# 경로 Top 4에서는 어떤 학벌이 많이 이용하는지 -> 연도합평균
route_p = route[route['access_path'].isin(['acquaintance', 'experience', 'no_information', 'internet_mobile_app'])]
route_sch = route_p.groupby('access_path')[["elmt","mid","high","univ+"]].mean()
route_sch

path = route_p['access_path'].unique()
index = np.arange(len(path))
bar_width=0.15

plt.clf()
plt.bar(index, route_sch["elmt"], bar_width, label="elmt")
plt.bar(index + bar_width, route_sch["mid"], bar_width, label="mid")
plt.bar(index + 2 * bar_width, route_sch["high"], bar_width, label="high")
plt.bar(index + 3 * bar_width, route_sch["univ+"], bar_width, label="univ+")

plt.rcParams['font.family'] ='Malgun Gothic'
plt.xlabel('여행 정보 획득 경로')
plt.title('여행 정보 획득 경로 Top 4 : 학벌',fontsize=13)
plt.xticks(index + bar_width, path, fontsize=6)
plt.legend(fontsize=7.8)
plt.show()



# TOP4 한번에 표현하기 - 연도별 성별로 다르게 표현 
##표1. top1 주변지인 추천
route2 = route[route['access_path'] == 'acquaintance']
route2 = route2.groupby('year')[['male', 'female']].mean()

plt.clf()

plt.subplot(2,2,1)
plt.figure(figsize=(12, 6))
route2.plot()
plt.xticks(route2.index, route2.index.astype(int))


##표2. top2 과거 경험
route3 = route[route['access_path'] == 'experience']
route3 = route3.groupby('year')[['male', 'female']].mean()

plt.subplot(2,2,2)
plt.figure(figsize=(12, 6))
route3.plot()
plt.xticks(route2.index, route2.index.astype(int))


##표3. top3 정보 없음
route4 = route[route['access_path'] == 'no_information']
route4 = route4.groupby('year')[['male', 'female']].mean()

plt.subplot(2,2,3)
plt.figure(figsize=(12, 6))
route4.plot()
plt.xticks(route2.index, route2.index.astype(int))


##표4. top4 인터넷 혹은 어플 활용
route5 = route[route['access_path'] == 'internet_mobile_app']
route5 = route5.groupby('year')[['male', 'female']].mean()

plt.subplot(2,2,4)
plt.figure(figsize=(12, 6))
route5.plot()
plt.xticks(route2.index, route2.index.astype(int))

plt.show()


# 경로 Top 3에서는 어떤 소득층이 많이 이용하는지 -> 연도합평균
## 1등-지인추천
route_exp = route[route['access_path'] == 'acquaintance']
route_exp = route_exp.groupby('year')[['l_sal', 'm_sal', 'h_sal']].mean()

years = route['year'].unique()
index = np.arange(len(years))
bar_width=0.2

plt.clf()

plt.bar(index, route_exp['l_sal'], bar_width, label='l_sal')
plt.bar(index + bar_width, route_exp['m_sal'], bar_width, label='m_sal')
plt.bar(index + 2 * bar_width, route_exp['h_sal'], bar_width, label='h_sal')

plt.rcParams['font.family'] ='Malgun Gothic'
plt.xlabel('년도')
plt.title('소득층에 따른 연도별 지인 추천 활용 비율',fontsize=13)
plt.xticks(index + bar_width, years)
plt.legend(fontsize=9)
plt.show()


## 2등-경험
route_exp = route[route['access_path'] == 'experience']
route_exp = route_exp.groupby('year')[['l_sal', 'm_sal', 'h_sal']].mean()

years = route['year'].unique()
index = np.arange(len(years))
bar_width=0.2

plt.clf()

plt.bar(index, route_exp['l_sal'], bar_width, label='l_sal')
plt.bar(index + bar_width, route_exp['m_sal'], bar_width, label='m_sal')
plt.bar(index + 2 * bar_width, route_exp['h_sal'], bar_width, label='h_sal')

plt.rcParams['font.family'] ='Malgun Gothic'
plt.xlabel('년도')
plt.title('소득층에 따른 연도별 과거 경험 활용 비율',fontsize=13)
plt.xticks(index + bar_width, years)
plt.legend(fontsize=9)
plt.show()


## 3등-정보없이
route_out = route[route['access_path'] == 'no_information']
route_out = route_out.groupby('year')[['l_sal', 'm_sal', 'h_sal']].mean()

years = route['year'].unique()
index = np.arange(len(years))
bar_width=0.2

plt.clf()

plt.bar(index, route_out['l_sal'], bar_width, label='l_sal')
plt.bar(index + bar_width, route_out['m_sal'], bar_width, label='m_sal')
plt.bar(index + 2 * bar_width, route_out['h_sal'], bar_width, label='h_sal')

plt.rcParams['font.family'] ='Malgun Gothic'
plt.xlabel('년도')
plt.title('소득층에 따른 연도별 여행 정보를 획득하지 않는 비율',fontsize=11)
plt.xticks(index + bar_width, years)
plt.legend(fontsize=7.5)
plt.show()


# 경로 순위
route.sort_values(["total"],ascending=False).head(5)
route.head(10)


