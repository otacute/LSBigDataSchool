# X ~ N(30, 5^2)
from scipy.stats import norm
import numpy as np

x = np.linspace(10, 50, 500)
pdf_values = norm.pdf(x, loc=30, scale=5)

import matplotlib.pyplot as plt

plt.plot(x, pdf_values)

# P(X>45) = ?
1 - norm.cdf(45, loc=30, scale=5)

# 상위 10% 값을 반환?
norm.ppf(0.9, loc=30, scale=5)

# X_bar 표본, n = 16
# X_bar ~ N(30, 5^2 / 16)
x = np.linspace(10, 50, 500)
x_bar_pdf_values = norm.pdf(x, loc=30, scale=np.sqrt((5**2)/16))
pdf_values = norm.pdf(x, loc=30, scale=5)
plt.plot(x, pdf_values)
plt.plot(x, x_bar_pdf_values, color="red")

# X_bar 표본, n = 16
# X_bar ~ N(30, 5^2 / 16)
# P(X_bar>38) = ? 
1 - norm.cdf(38, loc=30, scale=np.sqrt((5**2)/16))

# -------------------------------------------------------

# P(키트 양성|실제 양성)
# P(키트 양성 ∩ 실제 양성) / P(실제 양성)
import pandas as pd
covid = pd.DataFrame({
    "구분": ["키트 양성", "키트 음성"],
    "실제 양성": [370, 15],
    "실제 음성": [10, 690]
})
covid

# P(키트 양성|실제 양성)
# P(키트 양성 ∩ 실제 양성) / P(실제 양성)
covid.loc[0, '실제 양성'] / covid.loc[:, '실제 양성'].sum()

# P(실제 양성|키트 양성)
# P(실제 양성 ∩ 키트 양성) / P(키트 양성)
covid.loc[0, '실제 양성'] / covid[covid['구분']=='키트 양성'][['실제 양성','실제 음성']].sum(axis='columns')

# 코로나 발병률 = 0.01
# P(실제 양성) = 0.01 이란 의미
# P(실제 양성|키트 양성) = P(실제 양성 ∩ 키트 양성) / P(키트 양성) =  P(실제양성)P(키트양성|실제양성) / {P(실제양성)P(키트양성|실제양성) + P(실제음성)P(키트양성|실제음성)}
# a = P(실제양성)P(키트양성|실제양성)
# b = P(실제음성)P(키트양성|실제음성) = P(실제음성) * (P(키트양성∩실제음성)/P(실제음성))
# a / ( a + b )
a = 0.01 * (covid.loc[0, '실제 양성'] / covid.loc[:, '실제 양성'].sum())
b = 0.99 * (covid.loc[0, '실제 음성'] / covid.loc[:, '실제 음성'].sum())
prob = a / (a + b)
prob