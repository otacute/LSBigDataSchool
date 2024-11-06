from shiny.express import input, render, ui

# 20이 시작값
ui.input_slider("n", "숫자를 선택하세요!", 0, 100, 20)

@render.code
def txt():
    return f"n*2 is {input.n() * 2}"
