from shiny.express import input, render, ui

ui.page_opts(title="팔머펭귄 부리 깊이 예측하기!")

with ui.sidebar():
    ui.input_selectize(
        "var", " 펭귄 종을 선택하세요",
        choices=["Adelie", "Gentoo", "Chinstrap"])
    ui.input_slider('slider1', '부리길이를 입력하세요', min=0, max=100, value=50)

    @render.text
    def cal_depth():
        from palmerpenguins import load_penguins
        from sklearn.linear_model import LinearRegression
        import pandas as pd
        import numpy as np
        penguins = load_penguins().dropna()
        penguins_dummies = pd.get_dummies(penguins, 
                                columns=['species'],
                                drop_first=False)
        x = penguins_dummies[["bill_length_mm", "species_Chinstrap", "species_Gentoo"]]
        y = penguins_dummies['bill_depth_mm']

        model=LinearRegression()
        model.fit(x, y)
        input_df = pd.DataFrame({
            "bill_length_mm" : [input.slider1()],
            "species": pd.Categorical([input.var()],
                                      categories=["Adelie", "Chinstrap", "Gentoo"])
            
        })
        
        input_df = pd.get_dummies(
            input_df, 
            columns = ["species"],
            drop_first = True
        )
        y_hat = model.predict(input_df)
        y_hat = float(y_hat)
        return f'부리깊이 예상치: {y_hat:.2f}'



@render.plot # 데코레이터
def scatter():
    from palmerpenguins import load_penguins
    import seaborn as sns
    from matplotlib import pyplot as plt
    from sklearn.linear_model import LinearRegression
    import pandas as pd
    import numpy as np

    df = load_penguins()

    # 입력값: 펭귄 종, 부리 길이
    # 결과값: 부리 깊이
    # 선형회귀 모델 적합하기 문제

    model=LinearRegression()
    penguins=df.dropna()

    penguins_dummies = pd.get_dummies(penguins, 
                                columns=['species'],
                                drop_first=False)

    x = penguins_dummies[["bill_length_mm", "species_Chinstrap", "species_Gentoo"]]
    y = penguins_dummies['bill_depth_mm']

    model.fit(x, y)

    model.coef_
    model.intercept_

    regline_y=model.predict(x)

    index_1=np.where(penguins['species'] == "Adelie")
    index_2=np.where(penguins['species'] == "Gentoo")
    index_3=np.where(penguins['species'] == "Chinstrap")

    plt.rcParams.update({'font.family':'Malgun Gothic'})
    sns.scatterplot(data=df,
                    x='bill_length_mm',
                    y='bill_depth_mm',
                    hue='species')
    plt.plot(penguins["bill_length_mm"].iloc[index_1], regline_y[index_1], color="black")
    plt.plot(penguins["bill_length_mm"].iloc[index_2], regline_y[index_2], color="black")
    plt.plot(penguins["bill_length_mm"].iloc[index_3], regline_y[index_3], color="black")
    plt.xlabel('부리길이')
    plt.ylabel('부리깊이')