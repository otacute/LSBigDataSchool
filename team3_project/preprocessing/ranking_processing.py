import numpy as np
import pandas as pd

#데이터 파악하기
tour_2018 = pd.read_csv('data/ranking_2018.csv', encoding = 'cp949')
tour_2019 = pd.read_csv('data/ranking_2019.csv', encoding = 'cp949')
tour_2020 = pd.read_csv('data/ranking_2020.csv', encoding = 'cp949')
tour_2021 = pd.read_csv('data/ranking_2021.csv', encoding = 'cp949')
tour_2022 = pd.read_csv('data/ranking_2022.csv', encoding = 'cp949')
#전처리
# 열 이름 출력
column_names = tour_2018.columns.tolist()
print(column_names)

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

#top 30 추출하기
tour_2018 = tour_2018.sort_values(by = 'search_count', ascending = False).head(30)
tour_2018
tour_2019 = tour_2019.sort_values(by = 'search_count', ascending = False).head(30)
tour_2019
tour_2020 = tour_2020.sort_values(by = 'search_count', ascending = False).head(30)
tour_2020
tour_2021 = tour_2021.sort_values(by = 'search_count', ascending = False).head(30)
tour_2021
tour_2022 = tour_2022.sort_values(by = 'search_count', ascending = False).head(30)
tour_2022

tours = [tour_2018, tour_2019, tour_2020, tour_2021, tour_2022]

# concat 함수를 사용하여 데이터프레임들을 행 방향으로 합칩니다.
tour_top30 = pd.concat(tours, ignore_index=True)

# 합친 결과 확인
print(tour_top30)

#도로명 주소 열 삭제
tour_top30 = tour_top30.drop(columns=['address'])
tour_top30.info()

#순위 열 삭제
tour_top30.drop(columns=['ranking'], inplace=True)

#년도 행 추가하기
import pandas as pd
import numpy as np

# 반복할 년도 리스트
years = [2018, 2019, 2020, 2021, 2022]

# 각 년도에 해당하는 행 개수
rows_per_year = 30

# 각 년도에 대해 행 개수만큼 반복하여 'year' 리스트 생성
year_column = np.repeat(years, rows_per_year)

# 데이터프레임에 'year' 열 추가
tour_top30['year'] = year_column

# 결과 확인
print(tour_top30.head(40))  # 상위 40개 행 출력

# 시/군/구 동명으로 인해 추가 열 생성(혼선 막기 위함)
tour_top30['combined_city'] = tour_top30['state'] + " " +tour_top30['city']
tour_top30['combined_city'].head(10)

tour_top30.shape
tour_top30.to_excel('pre_tour_top30.xlsx', index=False)
<<<<<<< HEAD:sun_a/ranking_processing.py



# ccsscccsdfdfd
#dd
# 전처리한 데이터 파일 추출
# route.to_excel('pre_route.xlsx', index=False)
=======
>>>>>>> 9f3a96ff7e05ef0aea415c607acd64adbfda36f2:preprocessing/ranking_processing.py
