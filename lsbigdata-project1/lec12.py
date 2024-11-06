import pandas as pd

mpg = pd.read_csv("data/mpg.csv")
mpg.shape

# ! pip install seaborn
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(5,4)) # 사이즈 조정
sns.scatterplot(data=mpg,
                x = "displ", y = "hwy",
                hue="drv")\
    .set(xlim=[3,6], ylim=[10,30])
plt.show()
plt.clf()

# 막대 그래프
# mpg["drv"].unique()
df_mpg = mpg.groupby("drv", as_index=False)\
            .agg(mean_hwy = ('hwy','mean'))

df_mpg
sns.barplot(data=df_mpg.sort_values("mean_hwy"),
            x = "drv", y="mean_hwy",
            hue = "drv")

plt.show()
plt.clf()

# 교재 208p
# 빈도 막대 그래프 만들기

# 집단별 빈도표 만들기
df_mpg = mpg.groupby('drv', as_index=False)\
        .agg(n=('drv','count'))
        
df_mpg

# 막대 그래프 만들기
sns.barplot(data = df_mpg, x ='drv', y = 'n')
plt.show()
plt.clf()

# 빈도 막대 그래프
sns.countplot(data=mpg, x='drv')
plt.show()
plt.clf()
