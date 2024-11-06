# 교재 63p

# seaborn 패키지 설치
# ! pip install seaborn

import seaborn as sns
import matplotlib.pyplot as plt

var = ['a','a','b','c']
var

sns.countplot(x=var)
plt.show()
plt.clf()

df = sns.load_dataset("titanic")
sns.countplot(data=df, x="sex")
sns.countplot(data=df, x="sex", hue="sex")
# hue는 색을 의미하며, 데이터의 구분 기준을 정하여 색상을 통해 구분한다.
plt.show()
plt.clf()

sns.countplot(data=df,x="class")
plt.show()
plt.clf()

sns.countplot(data=df,x="class", hue="alive")
plt.show()
plt.clf()

?sns.countplot # help, 함수 설명
sns.countplot(data=df,x="class", hue="alive")
sns.countplot(data=df,x="class", hue="alive")
sns.countplot(data=df,
              x="class", 
              hue="alive", 
              orient="h") # orient 역할, data의 x, y 지정하지 않았을 때 방향 설정
plt.show()
plt.clf()

! pip install scikit-learn
import sklearn
# sklearn.metrics.accuracy_score()

from sklearn import metrics
# metrics.accuracy_score()

import sklearn.metrics as met 
# met.accuracy_score()
