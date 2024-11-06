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
df = df.drop(['15~19세', '20대', '30대', '40대', '50대', '60대', '70세 이상'], axis=1)
# 월급
df["l_sal"] = df["100만원 미만"] + df["100~200만원 미만"]
df["m_sal"] = df["200~300만원 미만"] + df["300~400만원 미만"] + df["400~500만원 미만"]
df["h_sal"] = df["500~600만원 미만"] + df["600만원 이상"]
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


