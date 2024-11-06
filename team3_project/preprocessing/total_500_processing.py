import numpy as np
import pandas as pd

# 데이터 로드
tour_2018 = pd.read_csv('data/ranking_2018.csv', encoding='cp949')
tour_2019 = pd.read_csv('data/ranking_2019.csv', encoding='cp949')
tour_2020 = pd.read_csv('data/ranking_2020.csv', encoding='cp949')
tour_2021 = pd.read_csv('data/ranking_2021.csv', encoding='cp949')
tour_2022 = pd.read_csv('data/ranking_2022.csv', encoding='cp949')

# 열 이름 변경
rename_dict = {'순위': 'ranking', '광역시/도': 'state', '시/군/구': 'city', '관광지명': 'spot',
               '도로명주소': 'address', '중분류 카테고리': 'category_m', '소분류 카테고리': 'category_s', '검색건수': 'search_count'}

tour_2018.rename(columns=rename_dict, inplace=True)
tour_2019.rename(columns=rename_dict, inplace=True)
tour_2020.rename(columns=rename_dict, inplace=True)
tour_2021.rename(columns=rename_dict, inplace=True)
tour_2022.rename(columns=rename_dict, inplace=True)

# 'cate_remove' 목록 정의
cate_remove = ['교통시설', '면세점', '백화점', '쇼핑몰', '대형마트', '기타쇼핑시설']

# 각 연도별 데이터에서 'cate_remove' 목록에 해당하는 카테고리를 제거
tour_2018 = tour_2018[~tour_2018['category_s'].isin(cate_remove)]
tour_2019 = tour_2019[~tour_2019['category_s'].isin(cate_remove)]
tour_2020 = tour_2020[~tour_2020['category_s'].isin(cate_remove)]
tour_2021 = tour_2021[~tour_2021['category_s'].isin(cate_remove)]
tour_2022 = tour_2022[~tour_2022['category_s'].isin(cate_remove)]

# 각 데이터프레임에 연도 열 추가
tour_2018['year'] = 2018
tour_2019['year'] = 2019
tour_2020['year'] = 2020
tour_2021['year'] = 2021
tour_2022['year'] = 2022

# 데이터 결합
tours = [tour_2018, tour_2019, tour_2020, tour_2021, tour_2022]
tour_total = pd.concat(tours, ignore_index=True)

# 불필요한 열 삭제
# 0722 지도 시각화를 위해 삭제 취소
# tour_total.drop(columns=['address', 'ranking'], inplace=True)

# 'combined_city' 열 추가
tour_total['combined_city'] = tour_total['state'] + " " + tour_total['city']

# 결과 저장
tour_total.to_excel('total_tour.xlsx', index=False)

# 결과 확인
tour_total.info()
print(tour_total.head(40))  # 상위 40개 행 출력
print(tour_total['category_s'].unique())  # 카테고리 확인
