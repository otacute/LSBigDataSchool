import numpy as np
2
2
import pandas as pd
3
3
import matplotlib.pyplot as plt
 
4
import seaborn as sns
4
5
5
 
route = pd.read_excel("C:/Users/USER/Documents/LS빅데이터스쿨/team3_project/younadata.xlsx")
 
6
 
7
route = pd.read_excel("C:/Users/USER/Documents/LS빅데이터스쿨/team3_project/youna_route.xlsx",header=1)
6
8
route.head(10)
7
9
8
10
# 변수 변경
9
 
route = route.rename(columns = {"시점":"year", "항목":"access_path", "소계" : "total", "남자" : "man", "여자" : "female"})
 
11
route = route.rename(columns = {"시점":"year", "항목":"access_path", "소계" : "total", "남자" : "male", "여자" : "female"})
10
12
11
13
route.head()
12
14
13
 
# 결측치 제거
14
 
route.replace('-', np.nan, inplace=True)
15
 
route["teens"].replace('-', np.nan, inplace=True)
16
 
route.isna().sum()
 
15
# 연도 결측치 처리
 
16
route.loc[0:8,"year"] = 2018
 
17
route.loc[9:17,"year"] = 2019
 
18
route.loc[18:26,"year"] = 2020
 
19
route.loc[27:35,"year"] = 2021
 
20
route.loc[36:43,"year"] = 2022
 
21
route["year"]=route["year"].astype(int)
 
22
route
17
23
18
24
# 항목
19
25
route.loc[route["access_path"]=="과거 방문 경험", "access_path"] = "experience"
 
 
@@ -27,12 +33,16 @@ route.loc[route["access_path"]=="정보 없이 방문", "access_path"] = "no_inf
27
33
route.loc[route["access_path"]=="기타", "access_path"] = "etc"
28
34
route.head()
29
35
 
36
# 결측치 제거
 
37
route.replace('-', np.nan, inplace=True)
 
38
route.isna().sum()
 
39
30
40
# 연령대
31
41
#route = route.rename(columns = {"15~19세":"teens", "20대":"20years", "30대" : "30years", 
32
42
#                                "40대":"40years", "50대":"50years", "60대":"60years",
33
43
#                               "70세 이상":"70years+"})
34
44
  
35
 
route["teens"] = df["15~19세"]                             
 
45
route["teens"] = route["15~19세"]   
36
46
route["young_adults"] = route["20대"] + route["30대"]
37
47
route["middle_adults"] = route["40대"] + route["50대"]
38
48
route["seniors"] = route["60대"] + route["70세 이상"]
 
 
@@ -69,14 +79,174 @@ route = route.drop(['임금봉급근로자', '고용원있는사업주', '고용
69
79
                    "자영업","전업주부","학생","무직은퇴","기타"], axis=1)
70
80
route
71
81
 
82
# 결측치 제거
 
83
route.replace('-', np.nan, inplace=True)
 
84
route.isna().sum()
 
85
72
86
# 전처리한 데이터 파일 추출
73
87
route.to_excel('pre_route.xlsx', index=False)
74
88
75
89
76
 
# 연도별 여행 정보 경로
77
 
travel.groupby("year")\
78
 
      .agg(mean_travel=("total", "mean"))
79
 
      
80
90
 
91
# 경로 평균 백분율 막대그래프
 
92
## 1. 5개년 평균 경로 순위 구하기
 
93
# route2["access_path"] = route["access_path"].drop_duplicates()
 
94
# route2= route.groupby("access_path").agg(total_mean=("total","mean")).reset_index()
 
95
# route2
 
96
 
97
#route2 = pd.DataFrame()
 
98
route2=route.groupby("access_path")\
 
99
.agg(total_mean = ("total","mean"))
 
100
route2
 
101
 
102
 
103
route2 = route2.sort_values(["total_mean"], ascending = False)
 
104
route2
 
105
 
106
## 2. 그래프화 화기
 
107
 
108
 
109
plt.clf()
 
110
plt.subplots_adjust(bottom=0.2)
 
111
sns.barplot(data=route2, x="access_path", y="total_mean", palette = "Set1")
 
112
plt.xlabel('여행 정보 획득 경로')
 
113
plt.ylabel('소계')
 
114
plt.xticks(rotation=20)
 
115
plt.xticks(fontsize=8)
 
116
plt.rcParams['font.family'] ='Malgun Gothic'
 
117
plt.title('5개년 평균 여행 정보 획득 경로',fontsize=15)
 
118
plt.show()
 
119
 
120
 
121
# 연도별로 정보 획득 경로 순위 어떻게 변하는지
 
122
route_path = route.groupby(['year', 'access_path'])['total'].mean().unstack()
 
123
plt.clf()
 
124
 
125
plt.figure(figsize=(12, 6))
 
126
route_path.plot()
 
127
 
128
plt.subplots_adjust(right=0.71)
 
129
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
 
130
plt.rcParams['font.family'] ='Malgun Gothic'
 
131
plt.title('연도 별 여행 정보 획득 경로',fontsize=15)
 
132
plt.show()
 
133
 
134
 
135
 
136
# 경로 Top 4에서는 어떤 가구수가 많이 이용하는지 -> 연도합평균
 
137
route_p = route[route['access_path'].isin(['acquaintance', 'experience', 'no_information', 'internet_mobile_app'])]
 
138
route_per = route_p.groupby('access_path')[["per1","per2","per3+"]].mean()
 
139
route_per
 
140
 
141
path = route_p['access_path'].unique()
 
142
index = np.arange(len(path))
 
143
bar_width=0.2
 
144
 
145
plt.clf()
 
146
plt.bar(index, route_per["per1"], bar_width, label="per1")
 
147
plt.bar(index + bar_width, route_per["per2"], bar_width, label="per2")
 
148
plt.bar(index + 2 * bar_width, route_per["per3+"], bar_width, label="per3+")
 
149
 
150
plt.rcParams['font.family'] ='Malgun Gothic'
 
151
plt.xlabel('여행 정보 획득 경로 Top 4')
 
152
plt.title('여행 정보 획득 경로 Top 4 : 가구원 수',fontsize=15)
 
153
plt.xticks(index + bar_width, path)
 
154
plt.legend()
 
155
plt.show()
 
156
 
157
 
158
 
159
 
160
route_per=route.groupby("access_path")\.agg(total_mean = ("total","mean"))
 
161
route_per
81
162
82
163
 
164
 
165
 
166
# 경로 Top 4에서는 어떤 성별이 많이 이용하는지 -> 연도합평균
 
167
 
168
 
169
 
170
 
171
 
172
# 경로 Top 4에서는 어떤 연령대가 많이 이용하는지 -> 연도합평균
 
173
 
174
 
175
 
176
 
177
 
178
 
179
 
180
# 경로 Top 3에서는 어떤 소득층이 많이 이용하는지 -> 연도합평균
 
181
## 1등-지인추천
 
182
route_exp = route[route['access_path'] == 'acquaintance']
 
183
route_exp = route_exp.groupby('year')[['l_sal', 'm_sal', 'h_sal']].mean()
 
184
 
185
years = route_exp['year'].unique()
 
186
index = np.arange(len(years))
 
187
bar_width=0.2
 
188
 
189
## 막대 그래프 그리기
 
190
plt.clf()
 
191
 
192
plt.bar(index, route_exp['l_sal'], bar_width, label='l_sal')
 
193
plt.bar(index + bar_width, route_exp['m_sal'], bar_width, label='m_sal')
 
194
plt.bar(index + 2 * bar_width, route_exp['h_sal'], bar_width, label='h_sal')
 
195
 
196
plt.xlabel('Year')
 
197
plt.ylabel('Salary')
 
198
plt.title('Yearly Travel Information Acquisition Rate by Income:Acquaintance',fontsize=10)
 
199
plt.xticks(index + bar_width, years)
 
200
plt.legend()
 
201
plt.show()
 
202
 
203
 
204
## 2등-경험
 
205
route_exp = route[route['access_path'] == 'experience']
 
206
route_exp = route_exp.groupby('year')[['l_sal', 'm_sal', 'h_sal']].mean()
 
207
 
208
years = route_exp['year'].unique()
 
209
index = np.arange(len(years))
 
210
bar_width=0.2
 
211
 
212
## 막대 그래프 그리기
 
213
plt.clf()
 
214
 
215
plt.bar(index, route_exp['l_sal'], bar_width, label='l_sal')
 
216
plt.bar(index + bar_width, route_exp['m_sal'], bar_width, label='m_sal')
 
217
plt.bar(index + 2 * bar_width, route_exp['h_sal'], bar_width, label='h_sal')
 
218
 
219
plt.xlabel('Year')
 
220
plt.ylabel('Salary')
 
221
plt.title('Yearly Travel Information Acquisition Rate by Income:Experience',fontsize=10)
 
222
plt.xticks(index + bar_width, years)
 
223
plt.legend()
 
224
plt.show()
 
225
 
226
 
227
## 3등-정보없이
 
228
route_exp = route[route['access_path'] == 'no_information']
 
229
route_exp = route_exp.groupby('year')[['l_sal', 'm_sal', 'h_sal']].mean()
 
230
 
231
years = route_exp['year'].unique()
 
232
index = np.arange(len(years))
 
233
bar_width=0.2
 
234
 
235
## 막대 그래프 그리기
 
236
plt.clf()
 
237
 
238
plt.bar(index, route_exp['l_sal'], bar_width, label='l_sal')
 
239
plt.bar(index + bar_width, route_exp['m_sal'], bar_width, label='m_sal')
 
240
plt.bar(index + 2 * bar_width, route_exp['h_sal'], bar_width, label='h_sal')
 
241
 
242
plt.xlabel('Year')
 
243
plt.ylabel('Salary')
 
244
plt.title('Yearly Travel Information Acquisition Rate by Income:No_information',fontsize=10)
 
245
plt.xticks(index + bar_width, years)
 
246
plt.legend()
 
247
plt.show()
 
248
 
249
 
250
# 경로 순위
 
251
route.sort_values(["total"],ascending=False).head(5)
 
252
route.head(10)
