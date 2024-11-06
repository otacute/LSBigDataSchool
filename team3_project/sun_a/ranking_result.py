import numpy as np
import pandas as pd

tour_top30 = pd.read_csv('pre_data/pre_tour_top30.csv', encoding = 'cp949')


import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# 광역시/도 빈도 확인하기
count_state = tour_top30['state'].value_counts()
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

# 시/군/구 구체적인 빈도 확인하기
count_combined_city = tour_top30['combined_city'].value_counts().head(10)
plt.figure(figsize=(3, 2))
count_combined_city.plot.bar(rot=45)
plt.xticks(fontsize=6)
plt.yticks(fontsize=8)
# 여백을 수동으로 조정합니다.
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
# 글자가 잘리지 않도록 레이아웃을 자동으로 조정합니다.
plt.tight_layout()
# 그림을 표시합니다.
plt.show()
plt.clf()

#관광지명 빈도 확인하기
count_city = tour_top30['spot'].value_counts()
plt.figure(figsize=(5, 3))
count_city.plot.bar(rot=80)
plt.xticks(fontsize=6)
plt.yticks(fontsize=8)
# 여백을 수동으로 조정합니다.
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
# 글자가 잘리지 않도록 레이아웃을 자동으로 조정합니다.
plt.tight_layout()
# 그림을 표시합니다.
plt.show()
plt.clf()

#중분류 빈도 확인하기
count_city = tour_top30['category_m'].value_counts()
plt.figure(figsize=(8, 4))
count_city.plot.bar(rot=45)
plt.xticks(fontsize=6)
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
count_city = tour_top30['category_s'].value_counts()
# 먼저 그림의 크기를 설정합니다.
plt.figure(figsize=(6, 4))
# 그런 다음 바 차트를 그립니다.
count_city.plot.bar(rot=45)
# x축과 y축 레이블 크기를 조정합니다.
plt.xticks(fontsize=6)
plt.yticks(fontsize=8)
# 여백을 수동으로 조정합니다.
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
# 글자가 잘리지 않도록 레이아웃을 자동으로 조정합니다.
plt.tight_layout()
# 그림을 표시합니다.
plt.show()
# 이후 그래프를 초기화합니다.
plt.clf()

print(tour_top30.columns)

tour_top30.sort_values('search_count', ascending = False)

# 연도별 요약하기
tour_top30.groupby('year') \
          .agg(search_count = ('search_count', 'mean'))
          
#관광지명 빈도 확인하기
count_city = tour_top30(['spot'] & ['year'] == 2018).value_counts()
plt.figure(figsize=(6, 3))
count_city.plot.bar(rot=45)
plt.xticks(fontsize=6)
plt.yticks(fontsize=8)
# 여백을 수동으로 조정합니다.
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
# 글자가 잘리지 않도록 레이아웃을 자동으로 조정합니다.
plt.tight_layout()
# 그림을 표시합니다.
plt.show()
plt.clf()

