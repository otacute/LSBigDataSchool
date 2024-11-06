import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 파일 불러오기
df = pd.read_excel("C:/Users/USER/Documents/LS빅데이터스쿨/LSBigdata_Project1/data/select.xlsx")
df.head()

# 변수 변경
df = df.rename(columns = {"소계" : "total", "남자" : "male", "여자" : "female"})
df.head()




# 변수 추가 및 삭제

# 연령대
df["teens"] = df["15~19세"]
df["young_adults"] = df["20대"] + df["30대"]
df["middle_adults"] = df["40대"] + df["50대"]
df["senior"] = df["60대"] + df["70세 이상"]
#2018
df.loc[0:14, "young_adults"] = (df.loc[0:14, "young_adults"] / df.loc[0:14, "young_adults"].sum() * 100).round(1)
df.loc[0:14, "middle_adults"] = (df.loc[0:14, "middle_adults"] / df.loc[0:14, "middle_adults"].sum() * 100).round(1)
df.loc[0:14, "senior"] = (df.loc[0:14, "senior"] / df.loc[0:14, "senior"].sum() * 100).round(1)
#2019
df.loc[15:29, "young_adults"] = (df.loc[15:29, "young_adults"] / df.loc[15:29, "young_adults"].sum() * 100).round(1)
df.loc[15:29, "middle_adults"] = (df.loc[15:29, "middle_adults"] / df.loc[15:29, "middle_adults"].sum() * 100).round(1)
df.loc[15:29, "senior"] = (df.loc[15:29, "senior"] / df.loc[15:29, "senior"].sum() * 100).round(1)
#2020
df.loc[30:44, "young_adults"] = (df.loc[30:44, "young_adults"] / df.loc[30:44, "young_adults"].sum() * 100).round(1)
df.loc[30:44, "middle_adults"] = (df.loc[30:44, "middle_adults"] / df.loc[30:44, "middle_adults"].sum() * 100).round(1)
df.loc[30:44, "senior"] = (df.loc[30:44, "senior"] / df.loc[30:44, "senior"].sum() * 100).round(1)
#2021
df.loc[45:59, "young_adults"] = (df.loc[45:59, "young_adults"] / df.loc[45:59, "young_adults"].sum() * 100).round(1)
df.loc[45:59, "middle_adults"] = (df.loc[45:59, "middle_adults"] / df.loc[45:59, "middle_adults"].sum() * 100).round(1)
df.loc[45:59, "senior"] = (df.loc[45:59, "senior"] / df.loc[45:59, "senior"].sum() * 100).round(1)
#2022
df.loc[60:74, "young_adults"] = (df.loc[60:74, "young_adults"] / df.loc[60:74, "young_adults"].sum() * 100).round(1)
df.loc[60:74, "middle_adults"] = (df.loc[60:74, "middle_adults"] / df.loc[60:74, "middle_adults"].sum() * 100).round(1)
df.loc[60:74, "senior"] = (df.loc[60:74, "senior"] / df.loc[60:74, "senior"].sum() * 100).round(1)


df = df.drop(['15~19세', '20대', '30대', '40대', '50대', '60대', '70세 이상'], axis=1)

# 월급
df["l_sal"] = df["100만원 미만"] + df["100~200만원 미만"]
df["m_sal"] = df["200~300만원 미만"] + df["300~400만원 미만"] + df["400~500만원 미만"]
df["h_sal"] = df["500~600만원 미만"] + df["600만원 이상"]
#2018
df.loc[0:14, "l_sal"] = (df.loc[0:14, "l_sal"] / df.loc[0:14, "l_sal"].sum() * 100).round(1)
df.loc[0:14, "m_sal"] = (df.loc[0:14, "m_sal"] / df.loc[0:14, "m_sal"].sum() * 100).round(1)
df.loc[0:14, "h_sal"] = (df.loc[0:14, "h_sal"] / df.loc[0:14, "h_sal"].sum() * 100).round(1)
#2019
df.loc[15:29, "l_sal"] = (df.loc[15:29, "l_sal"] / df.loc[15:29, "l_sal"].sum() * 100).round(1)
df.loc[15:29, "m_sal"] = (df.loc[15:29, "m_sal"] / df.loc[15:29, "m_sal"].sum() * 100).round(1)
df.loc[15:29, "h_sal"] = (df.loc[15:29, "h_sal"] / df.loc[15:29, "h_sal"].sum() * 100).round(1)
#2020
df.loc[30:44, "l_sal"] = (df.loc[30:44, "l_sal"] / df.loc[30:44, "l_sal"].sum() * 100).round(1)
df.loc[30:44, "m_sal"] = (df.loc[30:44, "m_sal"] / df.loc[30:44, "m_sal"].sum() * 100).round(1)
df.loc[30:44, "h_sal"] = (df.loc[30:44, "h_sal"] / df.loc[30:44, "h_sal"].sum() * 100).round(1)
#2021
df.loc[45:59, "l_sal"] = (df.loc[45:59, "l_sal"] / df.loc[45:59, "l_sal"].sum() * 100).round(1)
df.loc[45:59, "m_sal"] = (df.loc[45:59, "m_sal"] / df.loc[45:59, "m_sal"].sum() * 100).round(1)
df.loc[45:59, "h_sal"] = (df.loc[45:59, "h_sal"] / df.loc[45:59, "h_sal"].sum() * 100).round(1)
#2022
df.loc[60:74, "l_sal"] = (df.loc[60:74, "l_sal"] / df.loc[60:74, "l_sal"].sum() * 100).round(1)
df.loc[60:74, "m_sal"] = (df.loc[60:74, "m_sal"] / df.loc[60:74, "m_sal"].sum() * 100).round(1)
df.loc[60:74, "h_sal"] = (df.loc[60:74, "h_sal"] / df.loc[60:74, "h_sal"].sum() * 100).round(1)


df = df.drop(['100만원 미만', '100~200만원 미만', '200~300만원 미만', '300~400만원 미만', '400~500만원 미만',\
                '500~600만원 미만', '600만원 이상'], axis=1)
# 가구
df = df.rename(columns = { '1인' : 'per1',
			'2인' : 'per2',
			'3인이상' : 'per3+'})

# 학벌
df = df.rename(columns = { '초졸 이하' : 'elmt',
			 '중학교' : 'mid',
			'고등학교' : 'high',
			'대학교이상' : 'univ+'})
# 무응답 NR
df = df.rename(columns = {"무응답" : "nr",
                        "시점" : "year",
                        "항목" : "item"})
df.info()
df.columns

df.to_excel('pre_select.xlsx', index=False)




