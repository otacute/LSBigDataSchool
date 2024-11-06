# 패키지 불러오기
import numpy as np
import pandas as pd

tab3 = pd.read_csv('./data/tab3.csv')
tab3

tab1 = pd.DataFrame({"id":np.arange(1,13),
                    "score" : tab3["score"]})
tab1

tab2 = tab1.assign(gender =["female"] * 7  +["male"] * 5)
tab2

# 1 표본 t 검정(그룹 1개)
# 표본 t 검정 (그룹 1개)
# 귀무가설 vs. 대립가설
# H0 : mu = 10 vs Ha : mu != 10
# 유의수준 5%로 설정
from scipy.stats import ttest_1samp
result = ttest_1samp(tab1["score"], popmean=10, alternative='two-sided')
t_value = result[0] # t 검정통계량
p_value = result[1] # 유의확률 (p-value)
tab1["score"].mean() # 표본평균
result.pvalue
result.statistic
result.df
# 귀무가설이 참(mu=10)일 때, 11.53(표본평균)이 관찰될 확률이 6.48%이므로,
# 이것은 우리가 생각하는 보기 힘들다고 판단하는 기준인
# 0.05(유의수준)보다 크므로, 귀무가설이 거짓이라 판단하기 힘들다.
# 유의 확률 0.0648이 유의수준 0.05보다 크므로  
# 귀무가설을 기각하지 못한다.

# 95% 신뢰구간 구하기
ci = result.confidence_interval(confidence_level=0.95)
ci[0]
ci[1]

# 2 표본 t 검정(그룹 2개) - 분산 같고, 다를 때
# 분산 같은경우 : 독립 2표본 t검정
# 분산 다를경우 : 웰치스 t검정
# 귀무가설 vs. 대립가설
# H0 : mu_m = mu_f vs Ha : mu_m > mu_f
# 유의수준 1%로 설정, 두 그룹 분산 같다고 가정한다.
f_tab2 = tab2[tab2["gender"]=="female"]
m_tab2 = tab2[tab2["gender"]=="male"]

from scipy.stats import ttest_ind

result = ttest_ind(m_tab2["score"], f_tab2["score"],
                    equal_var=True, alternative='greater')
                    
# alternative = "less"의 의미는 대립가설이,
# 첫 번째 입력그룹의 평균이 두 번째 입력 그룹 평균보다 작다고 설정된 경우를 나타냄
# result = ttest_ind(f_tab2["score"], m_tab2["score"],
#                     equal_var=True, alternative='less')

result.statistic
result.pvalue
# ci = result.confidence_interval(confidence_level=0.95)
# ci[0]
# ci[1]

# 대응표본 t 검정(짝지을 수 있는 표본)
# 귀무가설 vs. 대립가설
# H0 : mu_before = mu_after vs Ha : mu_after > mu_before
# H0 : mu_d = 0 vs Ha : mu_d > 0
# mu_d = mu_after - mu_before
# 유의수준 1%로 설정

# mu_d에 대응하는 표본으로 변환
tab3_data = tab3.pivot_table(index='id', 
                             columns='group',
                             values='score').reset_index()
                             
tab3_data['score_diff'] = tab3_data['after'] - tab3_data['before']
test3_data = tab3_data[['score_diff']]
test3_data

from scipy.stats import ttest_1samp

result = ttest_1samp(tab3_data["score_diff"], popmean=0, alternative='greater')
t_value = result[0] # t 검정통계량
p_value = result[1] # 유의확률 (p-value)
tab1["score"].mean() # 표본평균

result.pvalue
result.statistic
result.df

#
# long to wide : pivot_table()
tab3_data = tab3.pivot_table(
    index='id', 
    columns='group',
    values='score'
    ).reset_index()


# wide to long : melt()
long_form = tab3_data.melt(
        id_vars='id', 
        value_vars=['before', 'after'], 
        var_name='group', 
        value_name='score'
        )
        
# 연습 1
df = pd.DataFrame({
    "id" : [1,2,3],
    "A"  : [10,20,30],
    "B"  : [40,50,60]
})
df

df_long = df.melt(id_vars="id",
            value_vars=["A","B"],
            var_name="group",
            value_name="score")
df_long

df_wide_1 = df_long.pivot_table(index = "id",
                    columns = "group",
                    values = "score")
df_wide_1
                    
df_wide_2 = df_long.pivot_table(
            columns = "group",
            values = "score")
df_wide_2

df_wide_3 = df_long.pivot_table(
            columns = "group",
            values = "score",
            aggfunc = "mean")
df_wide_3

df_wide_4 = df_long.pivot_table(
            columns = "group",
            values = "score",
            aggfunc = "max")
df_wide_4

# 연습 2
# import seaborn as sns
# tips = sns.load_dataset("tips")
# tips
# 
# tips.pivot_table(
#         columns = "day",
#         values = "tip")

# 요일별로 펼치고 싶은 경우
# index_list = list(tips.columns.delete(4))
# 
# tips.pivot_table(
#         index = ["index"],
#         columns = "day",
#         values = "tip").reset_index()
# 
# tips["day"].value_counts()

# 2024.08.07(수)
# 교재 Chapter 11. 지도 시각화
# 11-1. 시군구별 인구 단계 구분도 만들기
# JSON (JavaScript Object Notation)
import json
geo = json.load(open('./data/bigfile/SIG.geojson', encoding = 'UTF-8'))

# 행정 구역 코드 출력
geo['features'][0]['properties']

# 11-2. 서울시 동별 외국인 인구 단계 구분도 만들기
import json

geo_seoul = json.load(open('./data/bigfile/SIG_Seoul.geojson', encoding = 'UTF-8'))

# json - dict내에 dict가 있는 구조
type(geo_seoul)
len(geo_seoul)

geo_seoul.keys()
geo_seoul["features"]
len(geo_seoul["features"])
len(geo_seoul["features"][0])
type(geo_seoul["features"][0])
geo_seoul["features"][0].keys()

# 행정 구역 코드 출력
geo_seoul["features"][0]["properties"]
geo_seoul["features"][1]["properties"]

# 위도, 경도 좌표 출력
geo_seoul["features"][0]["geometry"]

coordinate_list = geo_seoul["features"][0]["geometry"]['coordinates']
coordinate_list
len(coordinate_list)
coordinate_list[0]
len(coordinate_list[0])
coordinate_list[0][0]
len(coordinate_list[0][0])

import numpy as np
coordinate_array = np.array(coordinate_list[0][0])
coordinate_array

x = coordinate_array[:,0]
y = coordinate_array[:,1]

import matplotlib.pyplot as plt

# 'Malgun Gothic' 폰트 적용
plt.rcParams['font.family'] = 'Malgun Gothic'
# 유니코드 마이너스 기호를 ASCII 하이픈으로 대체
plt.rcParams['axes.unicode_minus'] = False

# step을 통해 해상도 조절
plt.plot(x[::10], y[::10])
plt.show()
plt.clf()

# 함수로 만들기
def draw_seoul(num) :
    gu_name = geo_seoul["features"][num]["properties"]["SIG_KOR_NM"]
    coordinate_list = geo_seoul["features"][num]["geometry"]['coordinates']
    coordinate_array = np.array(coordinate_list[0][0])
    x = coordinate_array[:,0]
    y = coordinate_array[:,1]
    plt.title(gu_name)
    plt.rcParams.update({'font.family' : 'Malgun Gothic'})
    plt.plot(x, y)
    plt.show()
    plt.clf()
    
    return None

draw_seoul(0)

# 서울시 전체 지도 그리기
# gu_name | x | y
# ================
# 종로구  |126|36
# 종로구  |126|36
# 종로구  |126|36
# .........
# 종로구  |126|36
# 중구    |125|38
# 중구    |125|38
# 중구    |125|38
# .........


gu_name = [geo_seoul["features"][x]["properties"]["SIG_KOR_NM"] for x in range(len(geo_seoul["features"]))]
gu_name

coordinate_list = [geo_seoul["features"][x]["geometry"]['coordinates'] for x in range(len(geo_seoul["features"]))]
coordinate_list

np.array(coordinate_list[0][0][0])

# np.array(coordinate_list[0][0][0]) # 종로
# np.array(coordinate_list[1][0][0]) # 중구
# np.array(coordinate_list[2][0][0]) # 용산구
# np.array(coordinate_list[3][0][0]) # 성동구

import numpy as np
import pandas as pd

# 남규님 code - 종로구
pd.DataFrame({'gu_name' : [gu_name[0]] * len(np.array(coordinate_list[0][0][0])),
              'x'       : np.array(coordinate_list[0][0][0])[:,0],
              'y'       : np.array(coordinate_list[0][0][0])[:,1]})
              
# 한결 생각 - 얘는 왜인지 모르겠으나 syntax error 발생
# pd.DataFrame({'gu_name' : [gu_name[x]] * len(np.array(coordinate_list[x][0][0]) for x in range(len(geo_seoul["features"]))],
#               'x'       : [np.array(coordinate_list[x][0][0])[:,0] for x in range(len(geo_seoul["features"]))],
#               'y'       : [np.array(coordinate_list[x][0][0])[:,1]] for x in range(len(geo_seoul["features"])) })

# 빈 리스트 생성
empty = []

# for in 구문을 이용하하여 geo_seoul["features"]의 길이만큼 for 문 안의 내용을 반복
for x in range(len(geo_seoul["features"])):
    df = pd.DataFrame({
        'gu_name': [gu_name[x]] * len(np.array(coordinate_list[x][0][0])),
        'x': np.array(coordinate_list[x][0][0])[:, 0],
        'y': np.array(coordinate_list[x][0][0])[:, 1]
    })
    empty.append(df)

# 모든 DataFrame을 하나로 합치기, ignore_index=True를 이용하여 기존의 인덱스를 무시하고 새로운 인덱스 부여
seoul_total = pd.concat(empty, ignore_index=True)
seoul_total

import seaborn as sns
sns.scatterplot(data = seoul_total, x='x', y='y', hue="gu_name", s=5)
# plt.plot(x,y, hue="gu_name")
plt.show()
plt.clf()


# 서울시 전체 지도 그리기
# gu_name | x | y
# ================
# 종로구  |126|36
# 종로구  |126|36
# 종로구  |126|36
# .........
# 종로구  |126|36
# 중구    |125|38
# 중구    |125|38
# 중구    |125|38
# .........

# issac 선생님 code
# 방법 1
gu_name = list()
for i in range(25):
    gu_name.append(geo_seoul["features"][i]["properties"]["SIG_KOR_NM"])
gu_name

# 방법2
gu_name = [geo_seoul["features"][x]["properties"]["SIG_KOR_NM"] for x in range(len(geo_seoul["features"]))]
gu_name

# x, y 데이터 프레임
def make_seouldf(num) :
    gu_name = geo_seoul["features"][num]["properties"]["SIG_KOR_NM"]
    coordinate_list = geo_seoul["features"][num]["geometry"]['coordinates']
    coordinate_array = np.array(coordinate_list[0][0])
    x = coordinate_array[:,0]
    y = coordinate_array[:,1]

    return pd.DataFrame({"gu_name":gu_name,"x":x, "y":y})

make_seouldf(2)

# pd.concat([df_a, df_b])

# result = pd.DataFrame({
#     'gu_name' : [],
#     'x' : [],
#     'y' : []
# })

result = pd.DataFrame({})

for i in range(25) :
    result = pd.concat([result, make_seouldf(i)], ignore_index=True)
    
result

import seaborn as sns
sns.scatterplot(data = result, 
                x='x', y='y', 
                hue="gu_name", s=5,
                legend=False)
plt.show()
plt.clf()

# 강남과 강남이 아닌 지역을 구분해서 서울시 지도 그리기
result["gangnam"] = np.where(result["gu_name"] ==  "강남구", "강남", "안강남" )
result

sns.scatterplot(data = result, 
                x='x', y='y', 
                hue="gangnam", s=5,
                legend=False)

plt.show()
plt.clf()

# 서울 그래프 그리기 - palette 이용
import seaborn as sns
sns.scatterplot(data = result, 
                x='x', y='y', 
                hue="gu_name", s=5,
                legend=False,
                palette="deep")
plt.show()
plt.clf()

sns.scatterplot(data = result, 
                x='x', y='y', 
                hue="gu_name", s=5,
                legend=False,
                palette="viridis")
plt.show()
plt.clf()

# 서울 그래프 그리기 - 강남은 빨간색, 나머지는 회색
import seaborn as sns
gangnam_df = result.assign(is_gangnam = np.where(result["gu_name"] == "강남구", "강남","안강남"))
sns.scatterplot(data = gangnam_df, 
                x='x', y='y',
                palette={"안강남":"grey", "강남":"red"},
                hue="is_gangnam", s=5, )
plt.show()
plt.clf()

# 기본적으로 순서를 명시적으로 지정하지 않는다면 palette는 묵시적으로 unique순서대로 할당되는 듯
gangnam_df["is_gangnam"].unique()

import numpy as np
import pandas as pd
import json

geo_seoul = json.load(open('./data/bigfile/SIG_Seoul.geojson', encoding = 'UTF-8'))
geo_seoul['features'][0]['properties']

df_pop = pd.read_csv("data/Population_SIG.csv")
df_pop.head()
df_seoulpop = df_pop.iloc[1:26]
df_seoulpop["code"] = df_seoulpop["code"].astype(str)
# str로 타입을 변경해줘야 code를 그래프에 나타낼 수 있음
df_seoulpop.info()

# 패키지 설치하기
# !pip install folium
import folium

center_x = result["x"].mean()
center_y = result["y"].mean()
# p304
# 흰 도화지 맵 가져오기
map_sig = folium.Map(location=[37.551, 126.97], 
                    zoom_start=12,
                    tiles="cartodbpositron")
map_sig.save("map_seoul.html")

# Choropleth(코로플릿)사용해서 구 경계선 그리기

folium.Choropleth(
    geo_data = geo_seoul,
    data = df_seoulpop,
    columns=("code", "pop"),
    key_on = "feature.properties.SIG_CD").add_to(map_sig)

map_sig.save("map_seoul.html")

# 코로플릿 with bins
# matplotlib 팔레트
# tab10, tab20, Set1, Paired, Accent, Dark2, Pastel1, hsv 
# seaborn 팔레트
# deep, muted, bright, pastel, dark, colorblind, viridis, inferno, magma, plasma

bins = list(df_seoulpop["pop"].quantile([0, 0.2, 0.4, 0.6, 0.8, 1]))
bins

folium.Choropleth(
    geo_data = geo_seoul,
    data = df_seoulpop,
    columns = ('code', 'pop'),
    bins = bins,
    key_on = 'feature.properties.SIG_CD').add_to(map_sig)

map_sig.save('map_seoul.html')

folium.Choropleth(
    geo_data = geo_seoul,
    data = df_seoulpop,
    columns = ('code', 'pop'),
    fill_color ='viridis',
    bins = bins,
    key_on = 'feature.properties.SIG_CD').add_to(map_sig)

map_sig.save('map_seoul.html')

# 점 찍는 법
# make_seouldf(0).iloc[:,1:3].mean()
folium.Marker([37.583744,126.983800], popup="종로구").add_to(map_sig)
map_sig.save("map_seoul.html")

# 위도, 경도 - ames 표시
house_loc = pd.read_csv("data/houseprice/house_loc.csv")
house_loc = house_loc.iloc[:, -2:]

ll_mean = house_loc.mean()
Longitude_mean = ll_mean[0]
Latitude_mean = ll_mean[1]
ll_mean
Longitude_mean
Latitude_mean

map_ames = folium.Map(location=[Latitude_mean, Longitude_mean], 
                    zoom_start=12,
                    tiles="cartodbpositron")

map_ames.save("map_ames.html")

for i in range(len(house_loc)) : 
    folium.Marker([house_loc.iloc[i,1],house_loc.iloc[i,0]], popup=str(i)).add_to(map_ames)
    
map_ames.save("map_ames.html")

# -------------------------------------------------------------------------------
# 2024.08.07 (수) 오후 개인 프로젝트
# 하트 아이콘 표시가 가능할까?
heart_icon = """
<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 29.6">
    <path fill="blue" d="M23.6,0c-3.2,0-6,1.6-7.6,4.1C14.4,1.6,11.6,0,8.4,0C3.8,0,0,3.8,0,8.4C0,18.2,16,28.7,16,28.7s16-10.5,16-20.3
        C32,3.8,28.2,0,23.6,0z"/>
</svg>
"""

for i in range(len(house_loc)):
    icon = folium.DivIcon(html=f"""
        <div style="font-size:24px; color: blue;">
            {heart_icon}
        </div>""")
    folium.Marker(
        location=[house_loc.iloc[i, 1], house_loc.iloc[i, 0]],
        popup=str(i),
        icon=icon
    ).add_to(map_ames)

# map_ames를 HTML 파일로 저장합니다.
map_ames.save("map_ames.html")

# -------------------------------------------------------------------------------


