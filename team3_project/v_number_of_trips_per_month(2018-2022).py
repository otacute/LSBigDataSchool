# 사용 라이브러리 호출
import pandas as pd
import numpy as np 

# 전처리 된 데이터 불러오기
p_num_trip = pd.read_excel('pre_data/p_num_trip.xlsx')

# 데이터 정보 확인하기
# p_num_trip(년-월 별 여행 횟수)데이터 정보 확인하기
p_num_trip.info()
p_num_trip.shape
p_num_trip.head()
p_num_trip.describe()

# 분석을 용이하게 하기 위해 년도별 DataFrame생성
y2018 = p_num_trip.query('year == 2018')
y2019 = p_num_trip.query('year == 2019')
y2020 = p_num_trip.query('year == 2020')
y2021 = p_num_trip.query('year == 2021')
y2022 = p_num_trip.query('year == 2022')

# 데이터 시각화
# 시각화 라이브러리 호출
import matplotlib.pyplot as plt

# 그래프에 한글 표시하기
plt.rcParams['font.family'] ='Malgun Gothic'
# 언더바(_) 표시 가능하게 만듬
plt.rcParams['axes.unicode_minus'] =False

# 데이터 시각화

# 1. 2018년 ~ 2022년 기간동안 남녀의 여행 총 횟수를 원 그래프로 비교하기
sex_total = [p_num_trip['male'].sum(), p_num_trip['female'].sum()]
sex = ['남성','여성']

# 그래프 제목과 y축 레이블 설정
plt.title("2018 ~ 2022년 5년간 남녀의 총 여행 비율")

# 그래프 표시
# colors를 통해 남성은 파란색, 여성은 빨간색으로 매칭
# 비율을 계산해주는 autopct를 사용해 소수점 2째자리 까지 표시해주고 %를 나타내게 설정
# startangle을 사용하여 90도 부터 시작
plt.pie(sex_total, labels = sex, autopct = '%.2f%%',
        startangle=90, colors=['blue','red'])

plt.show()
plt.clf()

# 2. 2018년도 부터 2020년도 까지 남+여, 남, 여 여행 횟수 증감의 트랜드 구하기
# 데이터 선택

# for x in range 구문을 이용하여 다음과 같은 반복문을 실행
# sex_total2 = [y2018['total'].sum(), 
#               y2019['total'].sum(), 
#               y2020['total'].sum(), 
#               y2021['total'].sum(), 
#               y2022['total'].sum()]

sex_total2 = [x['total'].sum() for x in [y2018, y2019, y2020, y2021, y2022]]
male_total = [x['male'].sum() for x in [y2018, y2019, y2020, y2021, y2022]]
female_total = [x['female'].sum() for x in [y2018, y2019, y2020, y2021, y2022]]
years = [2018,2019,2020,2021,2022]

# 그래프 제목과 y축 레이블 설정
plt.title("2018 ~ 2022년 남녀의 총 여행 횟수")
plt.ylabel("단위 : 천회")
plt.xlabel("단위 : 년")

# 그래프 표시

x_range = np.arange(len(years))
plt.bar(x_range+0.0, sex_total2, width=0.25, label='총합', color = 'green')
plt.bar(x_range+0.3, male_total, width=0.25, label='남성', color = 'blue')
plt.bar(x_range+0.6, female_total, width=0.25, label='여성', color = 'red')

# x축 라벨 설정
plt.xticks(x_range+0.3, years)

# 범례추가
plt.legend()

# 그래프 보여주기
plt.show()
plt.clf()

# 3. 2018년도 부터 2020년도 까지 연령별 여행 횟수 증감의 트랜드 구하기
# 데이터 선택

# for x in range 구문을 이용하여 다음과 같은 반복문을 실행
teens_total = [x['teens'].sum() for x in [y2018, y2019, y2020, y2021, y2022]]
young_adults_total = [x['young_adults'].sum() for x in [y2018, y2019, y2020, y2021, y2022]]
middle_adults_total = [x['middle_adults'].sum() for x in [y2018, y2019, y2020, y2021, y2022]]
seniors_adults_total = [x['seniors'].sum() for x in [y2018, y2019, y2020, y2021, y2022]]
years = [2018,2019,2020,2021,2022]

# 그래프 제목과 y축 레이블 설정
plt.title("2018년 ~ 2022년 연령대 별 여행 횟수")
plt.ylabel("단위 : 천회")
plt.xlabel("단위 : 년")

# 그래프 표시
x_range = np.arange(len(years))
plt.bar(x_range+0.0, teens_total, width=0.2, label='청소년(15~19세)', color = 'lawngreen')
plt.bar(x_range+0.2, young_adults_total, width=0.2, label='청년(2~30대)', color = 'blue')
plt.bar(x_range+0.4, middle_adults_total, width=0.2, label='중년(4~50대)', color = 'red')
plt.bar(x_range+0.6, seniors_adults_total, width=0.2, label='노년(6~70대 이상)', color = 'gray')

# x축 라벨 설정
plt.xticks(x_range+0.3, years)

# 범례추가
plt.legend()

# 그래프 보여주기
plt.show()
plt.clf()

# 4. 가구원수 별 2018 ~ 2022년 연간 여행 횟수 비교
# for x in range 구문을 이용하여 다음과 같은 반복문을 실행
per1_total = [x['per1'].sum() for x in [y2018, y2019, y2020, y2021, y2022]]
per2_total = [x['per2'].sum() for x in [y2018, y2019, y2020, y2021, y2022]]
per3_total = [x['per3+'].sum() for x in [y2018, y2019, y2020, y2021, y2022]]
years = [2018,2019,2020,2021,2022]

# 그래프 제목과 y축 레이블 설정
plt.title("2018년 ~ 2022년 가구원 수 별 여행 횟수")
plt.ylabel("단위 : 천회")
plt.xlabel("단위 : 년")

# 그래프 표시
x_range = np.arange(len(years))
plt.bar(x_range+0.0, per1_total, width=0.2, label='1인 가구', color = 'slateblue')
plt.bar(x_range+0.2, per2_total, width=0.2, label='2인 가구', color = 'coral')
plt.bar(x_range+0.4, per3_total, width=0.2, label='3인 이상 가구', color = 'teal')

# x축 라벨 설정
plt.xticks(x_range+0.25, years) 

# 범례추가
plt.legend()

# 그래프 보여주기
plt.show()
plt.clf()

# 5. 가구 소득 별 2018 ~ 2022년 연간 여행 횟수 비교
# for x in range 구문을 이용하여 다음과 같은 반복문을 실행
l_sal_total = [x['l_sal'].sum() for x in [y2018, y2019, y2020, y2021, y2022]]
m_sal_total = [x['m_sal'].sum() for x in [y2018, y2019, y2020, y2021, y2022]]
h_sal_total = [x['h_sal'].sum() for x in [y2018, y2019, y2020, y2021, y2022]]
years = [2018,2019,2020,2021,2022]

# 그래프 제목과 y축 레이블 설정
plt.title("2018년 ~ 2022년 가구 소득 수 별 여행 횟수")
plt.ylabel("단위 : 천회")
plt.xlabel("단위 : 년")

# 그래프 표시
x_range = np.arange(len(years))
plt.bar(x_range+0.0, l_sal_total, width=0.2, label='저소득(200만원 미만)', color = 'thistle')
plt.bar(x_range+0.2, m_sal_total, width=0.2, label='중위소득(500만원 미만)', color = 'lightgreen')
plt.bar(x_range+0.4, h_sal_total, width=0.2, label='고소득(500만원 이상)', color = 'moccasin')

# x축 라벨 설정
plt.xticks(x_range+0.25, years)

# 범례추가
plt.legend()

# 그래프 보여주기
plt.show()
plt.clf()


# ?. 2018년 ~ 2022년 5년간 월별 여행 횟수 추이 변화 - 너무나도 안그려짐(다음에 혼자 해보기)
# 5 * 1 개의 그래프, 25 * 5 인치 크기 화면을 그리기 위해 subplots을 사용
# fig, ax = plt.subplots(5, 1, figsize=(25, 5))
# 격자 여백 설정
# plt.subplots_adjust(wspace = 0.5, hspace = 0.5)

# figure 전체 제목
# f.suptitle('월 별 여행 횟수')
# x_range_month = np.arange(1,13)
# ax[0, 0].bar[[x_range_month], [y2018['total']], label = "2018년"]
# ax[1, 0].bar[[x_range_month], [y2019['total']], label = "2019년"]
# ax[2, 0].bar[[x_range_month], [y2020['total']], label = "2020년"]
# ax[3, 0].bar[[x_range_month], [y2021['total']], label = "2021년"]
# ax[4, 0].bar[[x_range_month], [y2022['total']], label = "2022년"]

# 데이터 표시
# plt.show()
# plt.clf()

# 시험 삼아 해보기
# plt.bar(x_range_month, y2018['total'], width=0.25, label='총합', color = 'green')
# plt.ylim([20000, 31000])

# 데이터 표시
# plt.show()
# plt.clf()
