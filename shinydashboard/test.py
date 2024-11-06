from palmerpenguins import load_penguins
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
# ! pip install patsy
import patsy

df = load_penguins()
penguins=df.dropna()

model=LinearRegression()

# Patsy를 사용하여 수식으로 상호작용 항 생성
# 0 + 는 절편을 제거함, species와 island의 상호작용만 포함
# Y = a * X + b
# Y ~ X
# 종속변수 ~ 독립변수1 + 독립변수2
# formula = 'bill_depth_mm ~ bill_length_mm + species'
# formula = 'bill_depth_mm ~ 0 + bill_length_mm + species'
# formula = 'bill_depth_mm ~ 0 + bill_length_mm * species'
formula = 'bill_depth_mm ~ 0 + bill_length_mm + body_mass_g + flipper_length_mm + species'

y, x = patsy.dmatrices(formula, df,
                       return_type = "dataframe")
x.iloc[:, 1:]
model.fit(x, y)

model.coef_
model.intercept_

 