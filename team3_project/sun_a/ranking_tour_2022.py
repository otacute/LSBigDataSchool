import numpy as np
import pandas as pd

#데이터 파악하기
tour_2022 = pd.read_csv('data/ranking_2022.csv', encoding = 'cp949')
tour_2022.head()
tour_2022.info()
tour_2022.describe()
tour_2022.shape

#전처리
# 열 이름 출력
column_names = tour_2022.columns.tolist()
print(column_names)

tour_2022 = tour_2022.rename(columns = {'순위'      : 'ranking',    \
                                        '광역시/도' : 'state',       \
                                        '시/군/구'  : 'city',   \
                                        '관광지명'  : 'spot', \
                                        '도로명주소': 'address',    \
                                        '중분류 카테고리': 'category_m', \
                                        '소분류 카테고리': 'category_s', \
                                        '검색건수':        'search_count'
                                        
                                          })
                                
tour_2022
state = tour_2022['state'].unique()
print(state)

city = tour_2022['city'].unique()
print(city)
print(tour_2022['spot'].unique())
print(tour_2022['address'].unique())
print(tour_2022['category_m'].unique())
print(tour_2022['category_s'].unique())
print(tour_2022['search_count'].unique())

# 시/군/구 동명으로 인해 추가 열 생성(혼선 막기 위함)
tour_2022['combined_city'] = tour_2022['state'] + " " +tour_2022['city']
tour_2022.head(10)

import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# 광역시/도 빈도 확인하기
count_state = tour_2022['state'].value_counts()
plt.figure(figsize=(6, 3))
count_state.plot.bar(rot=65)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
# 여백을 수동으로 조정합니다.
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
# 글자가 잘리지 않도록 레이아웃을 자동으로 조정합니다.
plt.tight_layout()
# 그림을 표시합니다.
plt.show()
plt.clf()

# 시/군/구 빈도 확인하기
count_city = tour_2022['city'].value_counts().head(10)
plt.figure(figsize=(4, 2))
count_city.plot.bar(rot=45)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
# 여백을 수동으로 조정합니다.
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
# 글자가 잘리지 않도록 레이아웃을 자동으로 조정합니다.
plt.tight_layout()
# 그림을 표시합니다.
plt.show()
plt.clf()

# 시/군/구 구체적인 빈도 확인하기
count_combined_city = tour_2022['combined_city'].value_counts().head(10)
count_combined_city.plot.bar(rot=45)
plt.show()
plt.clf()

#관광지명 빈도 확인하기
count_city = tour_2022['spot'].value_counts().head(10)
plt.figure(figsize=(6, 3))
count_city.plot.bar(rot=45)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
# 여백을 수동으로 조정합니다.
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
# 글자가 잘리지 않도록 레이아웃을 자동으로 조정합니다.
plt.tight_layout()
# 그림을 표시합니다.
plt.show()
plt.clf()

#중분류 빈도 확인하기
count_city = tour_2022['category_m'].value_counts().head(10)
plt.figure(figsize=(8, 4))
count_city.plot.bar(rot=90)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
# 여백을 수동으로 조정합니다.
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
# 글자가 잘리지 않도록 레이아웃을 자동으로 조정합니다.
plt.tight_layout()
# 그림을 표시합니다.
plt.show()
# 이후 그래프를 초기화합니다.
plt.clf()


#소분류 빈도 확인하기
import matplotlib.pyplot as plt

# count_city를 계산합니다.
count_city = tour_2022['category_s'].value_counts().head(10)
# 먼저 그림의 크기를 설정합니다.
plt.figure(figsize=(6, 4))
# 그런 다음 바 차트를 그립니다.
count_city.plot.bar(rot=90)
# x축과 y축 레이블 크기를 조정합니다.
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
# 여백을 수동으로 조정합니다.
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
# 글자가 잘리지 않도록 레이아웃을 자동으로 조정합니다.
plt.tight_layout()
# 그림을 표시합니다.
plt.show()
# 이후 그래프를 초기화합니다.
plt.clf()



