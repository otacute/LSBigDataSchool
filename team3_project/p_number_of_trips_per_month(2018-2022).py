# 라이브러리 호출
import pandas as pd
import numpy as np

# 데이터 불러오기
# data : 원본 파일 저장 폴더
# pre_data : 전처리 후 파일 저장 폴더

# header를 두번째 행으로 고정
num_trip = pd.read_excel('data/R_월별_국내여행_횟수_20240714204514.xlsx', header=1)

# 데이터 전처리

# 열(columns)의 변수명 변경

# 시점
num_trip = num_trip.rename(columns = { '시점' : 'year', '항목' : 'month'})

# 전체
num_trip = num_trip.rename(columns = { '소계' : 'total'})

# 성별
num_trip = num_trip.rename(columns = { '남자' : 'male', '여자' :'female'})


# 학력
num_trip = num_trip.rename(columns = { '초졸 이하' : 'elmt', '중학교' : 'mid',
						               '고등학교' : 'high', '대학교이상' : 'univ+'})

# 가구원수
num_trip = num_trip.rename(columns = { '1인' : 'per1',
						               '2인' : 'per2',
						               '3인이상' : 'per3+'})

# 열(columns)의 파생변수 생성
# 연령 
# 15~19세, 20대, 30대, 40대, 50대, 60대, 70세 이상을 청소년, 청년, 중년, 노년 층으로 구분
num_trip["teens"] = num_trip["15~19세"]
num_trip["young_adults"] = num_trip["20대"] + num_trip["30대"]
num_trip["middle_adults"] = num_trip["40대"] + num_trip["50대"]
num_trip["seniors"] = num_trip["60대"] + num_trip["70세 이상"]

# 15~19세, 20대, 30대, 40대, 50대, 60대, 70세 이상 열 7개 삭제
num_trip = num_trip.drop(num_trip.columns[np.arange(5,12,1)], axis=1)

# 가구소득
# 저소득 ~ 200 만원 / 중위소득 200~500 만원 / 고소득 500~600이상 / 무응답
num_trip["l_sal"] = num_trip["100만원 미만"]     + num_trip["100~200만원 미만"]
num_trip["m_sal"] = num_trip["200~300만원 미만"] + num_trip["300~400만원 미만"] + num_trip["400~500만원 미만"]
num_trip["h_sal"] = num_trip["500~600만원 미만"] + num_trip["600만원 이상"]
num_trip["nr"]    = num_trip["무응답"]

# 100만원 미만, 100~200만원 미만, 200~300만원 미만, 300~400만원 미만,
# 400~500만원 미만, 500~600만원 미만, 600만원 이상, 무응답 총 8열 삭제
num_trip = num_trip.drop(num_trip.columns[np.arange(24,32,1)], axis=1)

# 열(columns) 삭제
# 직업 - 직업과 여행 횟수간의 상관관계가 크게 없다고 판단하여
# 임금봉급근로자, 고용원있는사업주, 고용원없는자영업자
# 무급가족 종사자, 사무전문, 기술생산노무, 판매서비스
# 자영업, 학생, 전업주부, 무직은퇴, 기타
# 12개의 열 삭제
num_trip = num_trip.drop(num_trip.columns[np.arange(5,17,1)], axis=1)

# 데이터 전처리 시
# 원본 : 39개 열, 파생변수 : 4 + 4 = 8 개열 생성, 열 삭제 : 27개 열 삭제
# 39 + 4 + 4 - 27 = 20 개의 columns이 생성됩니다.

# 데이터 확인
num_trip.info() 
num_trip.shape
num_trip.describe()
num_trip.head(12)
num_trip.tail(12)

# 데이터 문제점
# 1. year의 null값을 채워야함
# 2018, 2019, 2020, 2021, 2022 슬라이싱
num_trip['year'][1 :12] = 2018
num_trip['year'][13:24] = 2019
num_trip['year'][25:36] = 2020
num_trip['year'][37:48] = 2021
num_trip['year'][49:60] = 2022

# 2. month의 '월'값을 제거한 후 int값으로 형 변환 필요
# str 에서 맨 마지막 글자 제거
num_trip['month'] = num_trip['month'].str.replace('월','')
num_trip['month'] = num_trip['month'].astype('int')

# 3. nr(월급 미 응답자)의 nan값이 '-'로 대응되어 있음
num_trip.loc[num_trip["nr"] == "-", ["nr"]] = np.nan

# 데이터를 data폴더에 전처리 이후 데이터로 excel형태로 추출하고 형태 확인
# Unnamed: 0 이라는 인덱스가 생성되는 것을 방지하기 위해 index=False로 지정
# 가공된 데이터라는 의미인 (processing) 의 p를 파일 앞에 붙여 전처리 유무 구분
num_trip.to_excel(excel_writer = 'pre_data/p_num_trip.xlsx', index=False)
