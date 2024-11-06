from shiny.express import input, render, ui

ui.input_slider("val", "Slider label", min=0, max=100, value=50)

@render.text
def slider_val():
    return f"Slider value: {input.val()}"

# ---------------------


from shiny.express import input, render, ui

ui.input_selectize(
    "var", " 변수를 선택하세요",
    choices=["bill_length_mm", "body_mass_g", "bill_depth_mm"]
)

# @render.plot : 데코레이터
@render.plot
def hist():
    from matplotlib import pyplot as plt
    from palmerpenguins import load_penguins

    df = load_penguins()
    df[input.var()].hist(grid=False) # 인풋에서 선택한 var
    plt.xlabel(input.var()) 
    plt.ylabel("count")

