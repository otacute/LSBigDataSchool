# 라이브러리 호출
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import ccf
import matplotlib.dates as mdates
import scipy.stats as stats
from scipy.stats import ks_1samp, norm
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# 한글폰트 설정, 음수 부호 깨짐 방지
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
df1 = pd.read_csv("./프로젝트5주차/중합 CPS-8 DB D1.csv")
df2 = pd.read_csv("./프로젝트5주차/중합 CPS-8 DB D2.csv")

# 데이터 확인
df1.shape # 44641, 199
df2.shape # 44641, 213

# 데이터 결합
df = pd.merge(df1, df2)
df

# Timestamp 변수를 datetime 형식으로 변환
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# 데이터 확인
df.info()
df.describe()
df.head()
df.tail()
df.isna().sum().sum()

# 값이 한 개인 칼럼 찾기
df.columns[df.nunique() == 1]

df["LAB_8CHIP_TIO2 - Average"].unique() # 값이 1개
df["LAB_8CHIP_HEAT - Average"].unique() # 값이 1개
df["LAB_8CHIP_SIZE - Average"].unique() # 값이 1개
df["8PI_P4251A.PV - Average"].unique() # 값이 1개

# 값이 한 개인 칼럼 drop
df = df.drop(columns = ["LAB_8CHIP_TIO2 - Average", "LAB_8CHIP_HEAT - Average", "LAB_8CHIP_SIZE - Average", "8PI_P4251A.PV - Average"])

# 이상변수 확인하기
df["8SIC_P4360C.PV - Average"]

# 전체 기간의 이상 변수 그래프
sns.lineplot(data=df, x='Timestamp', y='8SIC_P4360C.PV - Average')

# 열 이름 변경
df.rename(columns={"8SIC_P4360C.PV - Average" : "방사C펌프RPM"}, inplace=True)

# 이상 시점 기준, 일주일 단위 뽑기 
df1 = df[(df['Timestamp'].dt.month == 4) & (df['Timestamp'].dt.day >= 2)].reset_index(drop = True)

# 4월 9일 00:00:01이 포함되어 제외
df1 = df1.drop(df1.index[-1])

# 데이터 확인
df1

# 이상 시점 기준, 2주, 3주, 4주 단위로 뽑고 저장하기

# 마지막 2주
df2 = df[(df['Timestamp'].dt.month >= 3) & (df['Timestamp'].dt.day >= 26) | (df['Timestamp'].dt.month == 4)].reset_index(drop = True)
df2 = df2.drop(df2.index[-1])
df2

# 마지막 3주
df3 = df[(df['Timestamp'].dt.month >= 3) & (df['Timestamp'].dt.day >= 19) | (df['Timestamp'].dt.month == 4)].reset_index(drop = True)
df3 = df3.drop(df3.index[-1])
df3

# 마지막 4주
df4 = df[(df['Timestamp'].dt.month >= 3) & (df['Timestamp'].dt.day >= 12) | (df['Timestamp'].dt.month == 4)].reset_index(drop = True)
df4 = df4.drop(df4.index[-1])
df4

# DataFrames 리스트
dataframes = [df1, df2, df3, df4]

# 1, 2, 3, 4주에 대한 차이를 알아보기 위해 모든 df에 대해 반복문 진행

# 반복 작업을 위한 루프 - 타겟 변수 기간 그래프
for idx, current_df in enumerate(dataframes, start=1):
    plt.figure(figsize=(14, 6))
    sns.lineplot(data=current_df, x='Timestamp', y='방사C펌프RPM')
    plt.title(f"Graph for 방사C펌프RPM in df{idx}")
    plt.xlabel("Timestamp")
    plt.ylabel("방사C펌프RPM")
    plt.grid(True)
    plt.show()

# 반복 작업을 위한 루프 - 숫자형, 카테고리, 오브젝트 타입 확인
for idx, current_df in enumerate(dataframes, start=1):
    # 변수들 타입 확인
    numeric_df = current_df.select_dtypes(include=['number'])
    categoric_df = current_df.select_dtypes(include=['category'])
    object_df = current_df.select_dtypes(include=['object'])
    
    # 출력
    print(f"\nData Type Analysis for df{idx}:")
    print(f"Numeric Variables: {numeric_df.shape[1]} columns")
    print(f"Categoric Variables: {categoric_df.shape[1]} columns")
    print(f"Object Variables: {object_df.shape[1]} columns")

# 반복 작업을 위한 루프 - 상관관계 시각화
for idx, current_df in enumerate(dataframes, start=1):
    # 상관 행렬 계산
    corr_matrix = current_df.corr()
    
    # 타겟 변수와의 상관 계수가 높은 Top 10 변수 선택
    target_variable = "방사C펌프RPM"
    top_10_corr = corr_matrix[target_variable].abs().sort_values(ascending=False).drop(target_variable).head(10)
    
    # 타겟 변수 포함한 리스트 생성
    top_10_vars = top_10_corr.index.tolist() + [target_variable]
    
    # 상관 행렬 시각화
    plt.figure(figsize=(10, 8))
    sns.heatmap(current_df[top_10_vars].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f"Correlation Matrix for Top 10 Correlated Variables with '{target_variable}' in {idx} week")
    plt.show()
    
# CCF 값 df 별로 출력
# for 문의 결과를 저장할 빈 dict 생성
results = {}

# 반복 작업을 위한 루프
for i, current_df in enumerate(dataframes, start=1):
    # 숫자형 데이터만 선택
    numeric_df = current_df.select_dtypes(include=['number'])
    
    # 타겟 변수 정의
    target_variable = "방사C펌프RPM"
    corr_matrix = current_df.corr()
    
    # 타겟 변수와의 상관계수가 높은 Top 10 변수 선택
    top_10_corr = corr_matrix[target_variable].abs().sort_values(ascending=False).drop(target_variable).head(10)
    top_10_vars = top_10_corr.index.tolist() + [target_variable]
    
    # CCF 분석 결과 저장
    lag_results = []
    target_series = current_df[target_variable]
    
    for var in top_10_vars:
        series = current_df[var]
        ccf_values = ccf(target_series, series, adjusted=False)
        
        # 최대 양의 상관계수와 최소(음의) 상관계수를 가지는 lag 찾기
        max_lag = np.argmax(ccf_values)
        min_lag = np.argmin(ccf_values)
        
        lag_results.append({
            'Variable': var,
            'Max Positive CCF Lag': max_lag,
            'CCF Value at Max Positive Lag': ccf_values[max_lag],
            'Max Negative CCF Lag': min_lag,
            'CCF Value at Max Negative Lag': ccf_values[min_lag]
        })
    
    # DataFrame으로 변환
    lag_df = pd.DataFrame(lag_results)
    print(lag_df)
    lag_df["CCF Lag"] = lag_df.apply(
        lambda row: row["Max Positive CCF Lag"] if abs(row["CCF Value at Max Positive Lag"]) >= abs(row["CCF Value at Max Negative Lag"]) else row["Max Negative CCF Lag"], axis=1
    )
    lag_df["CCF Value"] = lag_df.apply(
        lambda row: row["CCF Value at Max Positive Lag"] if abs(row["CCF Value at Max Positive Lag"]) >= abs(row["CCF Value at Max Negative Lag"]) else row["CCF Value at Max Negative Lag"], axis=1
    )
    
    # 최종 정리된 결과
    final_lag_df = lag_df[["Variable", "CCF Lag", "CCF Value"]]
    sorted_lag_df = final_lag_df.reindex(final_lag_df['CCF Value'].abs().sort_values(ascending=False).index)
    
    # 결과 저장
    results[f"df{i}_ccf_results"] = sorted_lag_df.reset_index(drop=True)

# 결과 출력 예시
results['df1_ccf_results']
results['df2_ccf_results']
results['df3_ccf_results']
results['df4_ccf_results']


# 위의 결과로 보아
# 일주일 이전의 변동은 의미없다고 판단해 df1만 사용하기로 함

###### df 1에 대한 변수 정리 #########################################

# 종속 변수 포함한 Top 3 변수 데이터 프레임
top_4_vars = results['df1_ccf_results'].head(4)

# 종속 변수 제외한 변수들 데이터프레임
top_10_vars = results['df1_ccf_results'].tail(10)

# 종속 변수 포함한 변수들 데이터프레임
top_11_vars = results['df1_ccf_results'].head(11)

# 종속 변수 제외한 변수들 리스트
top_10_var = top_10_vars['Variable'].tolist()

# 종속 변수 포함한 변수들 리스트
top_11_var = top_11_vars['Variable'].tolist() + [target_variable]

# 상관계수 상위 10개 변수의 교차 상관 그래프 그리기
# 2행 5열 레이아웃
fig, axs = plt.subplots(2, 5, figsize=(24, 10))

for i, var in enumerate(top_10_vars['Variable']):
    cross_corr = ccf(df1[target_variable].dropna(), df1[var].dropna(), adjusted=False)
    row, col = divmod(i, 5)  # 행, 열 위치 계산
    
    axs[row, col].stem(range(len(cross_corr)), cross_corr)
    axs[row, col].set_title(f"CCF between {target_variable} and {var}")
    axs[row, col].set_xlabel("Lag")
    axs[row, col].set_ylabel("CCF")
    axs[row, col].grid(True)

# 레이아웃 조정
plt.tight_layout()
plt.show()
  
    
# 교차 상관 subplot (상위 변수 3개 + 타겟 변수)

for idx, current_df in enumerate(dataframes, start=1):
    # 상위 4개의 변수 추출 (방사C펌프RPM 포함)
    top_4_vars = results[f'df{idx}_ccf_results'].head(4)
    variables = top_4_vars[top_4_vars['Variable'] != '방사C펌프RPM']['Variable'].values

    # Subplot 설정 (1x4 레이아웃)
    fig, axs = plt.subplots(4, 1, figsize=(12, 24), sharex=True)

    # 방사C펌프RPM 기본 시각화 (첫 번째 subplot)
    axs[0].plot(current_df['Timestamp'], current_df['방사C펌프RPM'], label='방사C펌프RPM', color='black', linewidth=2)
    axs[0].set_title("방사C펌프RPM")
    axs[0].set_xlabel("Timestamp")
    axs[0].set_ylabel("Values")
    axs[0].legend(loc='upper left')
    axs[0].grid(True)

    # 방사C펌프RPM을 제외한 각 변수에 대해 subplot 생성
    for i, var in enumerate(variables, start=1):
        axs[i].plot(current_df['Timestamp'], current_df[var], label=var)
        axs[i].set_title(var)
        axs[i].set_xlabel("Timestamp")
        axs[i].legend(loc='upper left')
        axs[i].grid(True)

    # 레이아웃 조정 및 표시
    plt.suptitle(f"Top 4 Variables with 방사C펌프RPM in df{idx}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
### 5분 단위 ##################################### 

# 교차 상관 plot (범위 더 좁혀서, 5분 간격으로)    

# 4월 8일부터 9시에서 15시까지 데이터만 필터링
april_8_data = df1[(df1['Timestamp'] >= pd.Timestamp("2019-04-08 09:00")) & (df1['Timestamp'] <= pd.Timestamp("2019-04-08 15:00"))]

# 상위 4개의 변수 추출 (방사C펌프RPM 포함)
top_4_vars = results['df1_ccf_results'].head(4)
variables = top_4_vars[top_4_vars['Variable'] != '방사C펌프RPM']['Variable'].values

# Subplot 설정 (1x4 레이아웃)
fig, axs = plt.subplots(4, 1, figsize=(12, 24), sharex=True)

# 방사C펌프RPM 기본 시각화 (첫 번째 subplot)
axs[0].plot(april_8_data['Timestamp'], april_8_data['방사C펌프RPM'], label='방사C펌프RPM', color='black', linewidth=2)
axs[0].set_title("방사C펌프RPM")
axs[0].set_ylabel("Values")
axs[0].legend(loc='upper left')
axs[0].grid(True)

# 방사C펌프RPM을 제외한 각 변수에 대해 subplot 생성
for i, var in enumerate(variables, start=1):
    axs[i].plot(april_8_data['Timestamp'], april_8_data[var], label=var)
    axs[i].set_title(var)
    axs[i].set_ylabel("Values")
    axs[i].legend(loc='upper left')
    axs[i].grid(True)

# x축을 4월 8일 09:00부터 15:00까지로 제한하고, 5분 간격으로 틱 설정
for ax in axs:
    ax.set_xlim(pd.Timestamp("2019-04-08 09:00"), pd.Timestamp("2019-04-08 15:00"))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))  # 5분 간격으로 표시
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # 포맷 설정

# 모든 subplot에 x축 레이블 추가
axs[-1].set_xlabel("Timestamp")

# 레이아웃 조정 및 표시
plt.suptitle("Top 4 Variables with 방사C펌프RPM from April 8, 9:00 to 15:00 in df1")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.xticks(rotation=45)  # x축 레이블 회전
plt.show()
  
  
# ======== 1분 단위 ======================

# 4월 8일부터 10시 필터링
april_8_data = df1[(df1['Timestamp'] >= pd.Timestamp("2019-04-08 10:00")) & (df1['Timestamp'] <= pd.Timestamp("2019-04-08 10:25"))]

# 상위 4개의 변수 추출 (방사C펌프RPM 포함)
top_4_vars = results['df1_ccf_results'].head(4)
variables = top_4_vars[top_4_vars['Variable'] != '방사C펌프RPM']['Variable'].values

# Subplot 설정 (1x4 레이아웃)
fig, axs = plt.subplots(4, 1, figsize=(12, 24), sharex=True)

# 방사C펌프RPM 기본 시각화 (첫 번째 subplot)
axs[0].plot(april_8_data['Timestamp'], april_8_data['방사C펌프RPM'], label='방사C펌프RPM', color='black', linewidth=2)
axs[0].set_title("방사C펌프RPM")
axs[0].set_ylabel("Values")
axs[0].legend(loc='upper left')
axs[0].grid(True)

# 방사C펌프RPM을 제외한 각 변수에 대해 subplot 생성
for i, var in enumerate(variables, start=1):
    axs[i].plot(april_8_data['Timestamp'], april_8_data[var], label=var)
    axs[i].set_title(var)
    axs[i].set_ylabel("Values")
    axs[i].legend(loc='upper left')
    axs[i].grid(True)

# x축을 4월 8일 10시부터 25분동안 1분 간격으로 설정
for ax in axs:
    ax.set_xlim(pd.Timestamp("2019-04-08 10:00"), pd.Timestamp("2019-04-08 10:25"))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))  # 1분 간격으로 표시
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # 포맷 설정

# 모든 subplot에 x축 레이블 추가
axs[-1].set_xlabel("Timestamp")

# 레이아웃 조정 및 표시
plt.suptitle("Top 4 Variables with 방사C펌프RPM from April 8, 9:00 to 15:00 in df1")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.xticks(rotation=45)  # x축 레이블 회전
plt.show()


# 각 변수에 대해 CCF 그래프 개별 출력
#for var in top_10_vars['Variable']:
    #cross_corr = ccf(df1[target_variable].dropna(), df1[var].dropna#(), adjusted=False)
    
    #plt.figure(figsize=(10, 6))
    #plt.stem(range(len(cross_corr)), cross_corr)
    #plt.title(f"CCF between {target_variable} and {var}")
    #plt.xlabel("Lag")
    #plt.ylabel("CCF")
    #plt.grid(True)
    #plt.show()
    
# df1 데이터에 대해 상위 3개의 변수와 방사C펌프RPM 시각화

# 방사C펌프RPM을 제외한 상위 3개 변수들로 시각화를 위해 필터링
variables = top_4_vars[top_4_vars['Variable'] != '방사C펌프RPM']['Variable'].values

# 4월 8일 09시 ~ 4월 8일 15시까지의 상위 3개의 변수와 방사C펌프RPM 시각화
plt.figure(figsize=(14, 8))

# 방사C펌프RPM을 기본으로 시각화
plt.plot(df1['Timestamp'], df1['방사C펌프RPM'], label='방사C펌프RPM', color='black', linewidth=2)

# 방사C펌프RPM을 제외한 상위 4개의 변수에 대해 시차 없이 원본 데이터로 시각화
for var in variables:
    plt.plot(df1['Timestamp'], df1[var], label=var)

# x축을 4월 8일 09시부터 4월 8일 15시까지로 제한
plt.xlim(pd.Timestamp("2019-04-08 09:00:00"), pd.Timestamp("2019-04-08 15:00:00"))

# 제목, 범례 및 축 설정
plt.title("Top 3 변수와 방사C펌프RPM의 변동 (4월 8일 09시 ~ 15시)")
plt.xlabel("Timestamp")
plt.ylabel("Values")
plt.legend(loc='upper left')
plt.grid(True)
plt.show()


# df1 데이터에 대해 상위 4개의 변수와 방사C펌프RPM 시각화 

# 4월 7일까지의 평균값 계산 (df 사용)
start_date = df['Timestamp'].min()
end_date = pd.Timestamp("2019-04-07")
average_values = {}

# 각 변수에 대해 4월 7일까지의 평균 계산
for var in ['방사C펌프RPM'] + list(variables):
    average_values[var] = df[(df['Timestamp'] >= start_date) & (df['Timestamp'] <= end_date)][var].mean()

# Subplot 설정 (4x1 레이아웃)
fig, axs = plt.subplots(4, 1, figsize=(12, 24))

# 방사C펌프RPM 기본 시각화 (첫 번째 subplot에만 방사C펌프RPM 추가)
axs[0].plot(df1['Timestamp'], df1['방사C펌프RPM'], label='방사C펌프RPM', color='black', linewidth=2)
axs[0].axhline(y=average_values['방사C펌프RPM'], color='red', linestyle='--', linewidth=1.5, label='평균선 (3월 9일부터 4월 7일까지)')
axs[0].set_title("방사C펌프RPM((8SIC_P4360C.PV))")
axs[0].set_xlabel("시간")
axs[0].set_ylabel("Values")
axs[0].legend(loc='upper left')
axs[0].grid(True)

# 방사C펌프RPM을 제외한 각 변수에 대해 subplot 생성
for i, var in enumerate(variables, start=1):
    axs[i].plot(df1['Timestamp'], df1[var], label=var)
    axs[i].axhline(y=average_values[var], color='red', linestyle='--', linewidth=1.5, label='평균선 (3월 9일부터 4월 7일까지)')
    axs[i].set_title(var)
    axs[i].set_xlabel("시간")
    axs[i].set_ylabel("Values")
    axs[i].legend(loc='upper left')
    axs[i].grid(True)

# x축을 4월 8일로 제한하고 시간 포맷을 설정 (1시간 단위)
for ax in axs:
    ax.set_xlim(pd.Timestamp("2019-04-08"), df1['Timestamp'].max())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))

# 레이아웃 조정 및 표시
plt.suptitle("4월 8일의 주요변수와 방사C펌프RPM 변동 (4월 7일까지 평균선 포함)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# 구간을 더 자세히 보기 위해 df1에서 x축을 더 줄인 뒤, 기존 값과 lag shift 했을 때 비교

### Lag 조정 전 #####################################

# 데이터가 시간 형식으로 되어 있으니, 필터링 범위 설정
start_time = pd.Timestamp("2019-04-08 09:00:00")
end_time = pd.Timestamp("2019-04-08 14:00:00")
lag_value = 0  # 설정된 lag 값

# 데이터 필터링
filtered_df = df1[(df1['Timestamp'] >= start_time) & (df1['Timestamp'] <= end_time)]

# 시각화를 위한 설정
plt.figure(figsize=(14, 8))

# 타겟 변수 방사C펌프RPM 시각화
plt.plot(filtered_df['Timestamp'], filtered_df['방사C펌프RPM'], label='방사C펌프RPM', color='black', linewidth=2)

# 8PI_P4362C.PV - Average 시각화 (lag 적용)
shifted_series = filtered_df['8PI_P4362C.PV - Average'].shift(lag_value)
plt.plot(filtered_df['Timestamp'], shifted_series, label=f'8PI_P4362C.PV - Average (lag: {lag_value})', color='blue')

# 제목, 범례 및 축 설정
plt.title("8PI_P4362C.PV - Average (lag 적용)와 방사C펌프RPM의 변동 (4월 8일 9시~14시)")
plt.xlabel("Timestamp")
plt.ylabel("Values")
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

### Lag 조정 후 #########################################

# 데이터가 시간 형식으로 되어 있으니, 필터링 범위 설정
start_time = pd.Timestamp("2019-04-08 09:00:00")
end_time = pd.Timestamp("2019-04-08 14:00:00")
lag_value = -5  # 설정된 lag 값

# 데이터 필터링
filtered_df = df1[(df1['Timestamp'] >= start_time) & (df1['Timestamp'] <= end_time)]

# 시각화를 위한 설정
plt.figure(figsize=(14, 8))

# 타겟 변수 방사C펌프RPM 시각화
plt.plot(filtered_df['Timestamp'], filtered_df['방사C펌프RPM'], label='방사C펌프RPM', color='black', linewidth=2)

# 8PI_P4362C.PV - Average 시각화 (lag 적용)
shifted_series = filtered_df['8PI_P4362C.PV - Average'].shift(lag_value)
plt.plot(filtered_df['Timestamp'], shifted_series, label=f'8PI_P4362C.PV - Average (lag: {lag_value})', color='blue')

# 제목, 범례 및 축 설정
plt.title("8PI_P4362C.PV - Average (lag 적용)와 방사C펌프RPM의 변동 (4월 8일 9시~14시)")
plt.xlabel("Timestamp")
plt.ylabel("Values")
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

## 탑 10 변수들 산점도 시각화

# 총 변수 개수와 한 페이지당 변수 개수 설정
num_vars = len(top_10_var)  # 실제 top 10 변수로 변경
vars_per_page = 5

# 페이지 수 계산
num_pages = (num_vars + vars_per_page - 1) // vars_per_page  # 올림

# 각 페이지별로 산점도 생성
for page in range(num_pages):
    start_index = page * vars_per_page
    end_index = start_index + vars_per_page
    variables = top_10_var[start_index:end_index]
    
    # 페이지당 산점도 생성
    plt.figure(figsize=(15, 20))
    for i, var in enumerate(variables, start=1):
        plt.subplot(vars_per_page, 1, i)
        plt.scatter(df[var], df[target_variable], alpha=0.5)
        plt.title(f'{target_variable} vs {var} 산점도')
        plt.xlabel(var)
        plt.ylabel(target_variable)
    
    plt.tight_layout()
    plt.show()


# 독립변수 정규성 파악

### 모든 변수에 대해 히스토그램과 Q-Q 플롯을 생성 ################

# 각 변수에 대해 히스토그램과 Q-Q 플롯을 생성 (10개의 세트를 2행 5열 레이아웃으로 표시)
fig, axs = plt.subplots(5, 4, figsize=(20, 25))  # 5행 4열 레이아웃으로 총 20개의 그래프

for i, var in enumerate(top_10_var):
    # 히스토그램을 왼쪽에 배치
    axs[i // 2, (i % 2) * 2].hist(df1[var].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axs[i // 2, (i % 2) * 2].set_title(f"{var} - 히스토그램")
    axs[i // 2, (i % 2) * 2].set_xlabel("값")
    axs[i // 2, (i % 2) * 2].set_ylabel("빈도수")

    # Q-Q 플롯을 오른쪽에 배치
    stats.probplot(df1[var].dropna(), dist="norm", plot=axs[i // 2, (i % 2) * 2 + 1])
    axs[i // 2, (i % 2) * 2 + 1].set_title(f"{var} - Q-Q Plot")

plt.tight_layout()
plt.show()

### Top 3 변수에 대해 히스토그램과 Q-Q 플롯을 생성 ###############

# Top 3 변수 재설정
variables = top_4_vars[top_4_vars['Variable'] != '방사C펌프RPM']['Variable'].values

fig, axs = plt.subplots(3, 2, figsize=(14, 12))  # 3행 2열 레이아웃으로 총 6개의 그래프

for i, var in enumerate(variables):
    # 히스토그램을 왼쪽에 배치
    axs[i, 0].hist(df1[var].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axs[i, 0].set_title(f"{var} - 히스토그램")
    axs[i, 0].set_xlabel("값")
    axs[i, 0].set_ylabel("빈도수")

    # Q-Q 플롯을 오른쪽에 배치
    stats.probplot(df1[var].dropna(), dist="norm", plot=axs[i, 1])
    axs[i, 1].set_title(f"{var} - Q-Q Plot")

plt.tight_layout()
plt.show()

# 각 변수에 대해 Kolmogorov-Smirnov 검정(정규성 검정)을 수행
# 영가설(귀무가설) : 데이터가 정규분포를 따른다
# p-value가 0.05보다 낮으면 영가설을 기각하기에 
# p-value가 높아야 정규분포를 따른다는 것

# 상위 10개 변수에 대한 Kolmogorov-Smirnov 검정
ks_results_top_10 = {}

for var in top_10_var:
    # NaN 값을 제거하고 정규 분포와 비교하여 KS 검정 수행
    stat, p_value = ks_1samp(df1[var].dropna(), norm.cdf, args=(df1[var].mean(), df1[var].std()))
    ks_results_top_10[var] = {'Statistic': stat, 'P-Value': p_value}

# 결과를 데이터프레임으로 정리
ks_df_top_10 = pd.DataFrame(ks_results_top_10).T

# 결과 출력
print("Top 10 Variables KS Test Results")
print(ks_df_top_10)

# 상관계수 Top 3 변수에 대해서만 Kolmogorov-Smirnov 검정 진행
ks_results_top_3 = {}

for var in variables:
    # NaN 값을 제거하고 정규 분포와 비교하여 KS 검정 수행
    stat, p_value = ks_1samp(df1[var].dropna(), norm.cdf, args=(df1[var].mean(), df1[var].std()))
    ks_results_top_3[var] = {'Statistic': stat, 'P-Value': p_value}

# 결과를 데이터프레임으로 정리
ks_df_top_3 = pd.DataFrame(ks_results_top_3).T

# 결과 출력
print("\nTop 3 Variables KS Test Results")
print(ks_df_top_3)

# ks_df_top_10과 ks_df_top_3 출력 (데이터프레임 형태)
print("Top 10 Variables KS Test DataFrame")
display(ks_df_top_10)

print("\nTop 3 Variables KS Test DataFrame")
display(ks_df_top_3)

# 모든 변수의 P-Value가 낮아 영가설을 기각하므로, 정규성이 없다.