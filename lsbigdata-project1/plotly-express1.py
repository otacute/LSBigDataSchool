# 2024. 08. 12(월)-------------------------------------------------
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head()

# x: bill_length_mm
# y: bill_depth_mm  
fig = px.scatter(
    penguins,
    x="bill_length_mm",
    y="bill_depth_mm",
    color="species",
    # trendline="ols" # p134.
)
fig.show()

fig.update_layout(
    title={'text' : "팔머펭귄",
           'x': 0.5, 
           'xanchor' : "center",
           'y' : 0.5}
)
fig.show()

# title을 굵게, 색을 파란색으로 지정 - 내 방법
fig.update_layout(
    title={'text' : "<span style='font-weight:bold'>\
           <span style = 'color:blue'>팔머펭귄</span></span>",
           'x': 0.5, 
           'xanchor' : "center",
           'y' : 0.5}
)
fig.show()

# title을 굵게, 색을 파란색으로 지정 - 상후 방법
fig.update_layout(
    title={'text' : "<span style='color:blue;font-weight:bold'>팔머펭귄</span>",
           'x': 0.5, 
           'xanchor' : "center",
           'y' : 0.5}
)
fig.show()


# CSS 문법 이해하기
# <span>
#     <span style='font-weight:bold'> ... </span>
#     <span> ... </span>
#     <span> ... </span>
# </span>

