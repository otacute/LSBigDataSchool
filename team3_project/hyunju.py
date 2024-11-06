import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# 한글 설정
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

df = pd.read_excel("./pre_data/pre_select.xlsx")
df

df_2018 = df.loc[0:14]
df_2019 = df.loc[15:29]
df_2020 = df.loc[30:44]
df_2021 = df.loc[45:59]
df_2022 = df.loc[60:74]
df_2018.dtypes()

total_2018 = df_2018.sort_values("total", ascending = False).head()
total_2019 = df_2019.sort_values("total", ascending = False).head()
total_2020 = df_2020.sort_values("total", ascending = False).head()
total_2021 = df_2021.sort_values("total", ascending = False).head()
total_2022 = df_2022.sort_values("total", ascending = False).head()

total_all = pd.concat([total_2018, total_2019, total_2020, total_2021, total_2022]).iloc[:, :3]
total_all_pivot = total_all.pivot(index='year', columns='item', values='total')
total_all_pivot.plot(kind='bar', figsize=(10, 6))

fig = px.bar(total_all_pivot, title='년도별 여행지 선택 5순위', labels={'value': '총합', 'year': '년도'})
fig.update_layout(
    xaxis_title='Year',
    yaxis_title='Total',
    legend_title_text='Item',
    legend=dict(bbox_to_anchor=(1.05, 1), x=1)
)
fig.update_xaxes(tickangle=0)  # x축 레이블을 수평으로 설정

fig.show()


# plt.title('년도별 여행지 선택 5순위')
# plt.xlabel('Year')
# plt.ylabel('Total')
# plt.legend(title='Item', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.xticks(rotation=0)  # x축 레이블을 수평으로 설정
# plt.show()
# plt.clf()
---------------------------------------------------------------------------------

#선 그래프
visualappeal = total_all.query("item == '볼거리 제공'")
Popularity = total_all.query("item == '여행지 지명도'")
time = total_all.query("item == '여행할 수 있는 시간'")


# 선 그래프 생성
plt.figure(figsize=(10, 6))
plt.plot(visualappeal["year"], visualappeal["total"], marker='o',label="볼거리 제공")
plt.plot(Popularity["year"], Popularity["total"], marker='o',label="여행지 지명도")
plt.plot(time["year"], time["total"], marker='o',label="여행할 수 있는 시간")


# 그래프에 제목과 축 레이블 추가
plt.title("여행을 선택하는 이유 - 전체")
plt.xlabel("년도")
plt.ylabel("전체 %")
plt.legend(fontsize='x-small',loc='center left')
plt.show()
---------------------------------------------------------------------------------
# 남녀의 여행 선택 차이

#남자
male_2018 = df_2018[["item", "male"]].sort_values(by="item")
male_2019 = df_2019[["item", "male"]].sort_values(by="item")
male_2020 = df_2020[["item", "male"]].sort_values(by="item")
male_2021 = df_2021[["item", "male"]].sort_values(by="item")
male_2022 = df_2022[["item", "male"]].sort_values(by="item")

male_All = pd.merge(male_2018, male_2019, how = 'left', on = 'item')
male_All = pd.merge(male_All, male_2020, how = 'left', on = 'item', suffixes=('', '_2020'))
male_All = pd.merge(male_All, male_2021, how = 'left', on = 'item', suffixes=('', '_2021'))
male_All = pd.merge(male_All, male_2022, how = 'left', on = 'item', suffixes=('', '_2022'))


# 각 연도별 열 이름 변경 (이미 올바르게 되어있다면 생략 가능)
male_All.columns = ['item', 'male_2018', 'male_2019', 'male_2020', 'male_2021', 'male_2022']

# 각 행의 총합을 계산하여 새로운 열에 추가합니다.
male_All['total_male'] = male_All[['male_2018', 'male_2019', 'male_2020', 'male_2021', 'male_2022']].sum(axis=1)

# 필요 없는 열 삭제 및 total_male 기준으로 정렬
male_All = male_All.drop(columns=['male_2018', 'male_2019', 'male_2020', 'male_2021', 'male_2022']).sort_values(by='total_male', ascending=False)

male_All.head()

# 여자 데이터 정렬
female_2018 = df_2018[["item", "female"]].sort_values(by="item")
female_2019 = df_2019[["item", "female"]].sort_values(by="item")
female_2020 = df_2020[["item", "female"]].sort_values(by="item")
female_2021 = df_2021[["item", "female"]].sort_values(by="item")
female_2022 = df_2022[["item", "female"]].sort_values(by="item")

# 데이터 병합
female_All = pd.merge(female_2018, female_2019, how='left', on='item', suffixes=('_2018', '_2019'))
female_All = pd.merge(female_All, female_2020, how='left', on='item')
female_All = pd.merge(female_All, female_2021, how='left', on='item')
female_All = pd.merge(female_All, female_2022, how='left', on='item')

# 각 연도별 열 이름 변경 (필요한 경우)
female_All.columns = ['item', 'female_2018', 'female_2019', 'female_2020', 'female_2021', 'female_2022']

# 각 행의 총합을 계산하여 새로운 열에 추가합니다.
female_All['total_female'] = female_All[['female_2018', 'female_2019', 'female_2020', 'female_2021', 'female_2022']].sum(axis=1)

# 필요 없는 열 삭제 및 total_female 기준으로 내림차순 정렬
female_All = female_All.drop(columns=['female_2018', 'female_2019', 'female_2020', 'female_2021', 'female_2022']).sort_values(by='total_female', ascending=False)

F =female_All.head().plot.bar(rot = 0, color='blue', title='여행지 고르는 이유 순위 - 여자')
F.set_xticklabels(['볼거리', '지명도', '여행시간', '이동거리', '음식'])

plt.show()
plt.clf()

M = male_All.head().plot.bar(rot = 0, color='green', title='여행지 고르는 이유 순위 - 남자')
M.set_xticklabels(['볼거리', '지명도', '이동거리', '여행시간', '동반자유형'])

plt.show()
plt.clf()

--------------------------------------------------------------------------------
# 세대별 선호하는 이유 

























