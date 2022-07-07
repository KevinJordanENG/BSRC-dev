using PlotlyJS

tool = ["turboSETI (CPU)", "hyperSETI", "turboSETI (GPU)", "SETIcore"]
val =[22.1, 5.56, 3.31, 0.9]
plot(bar(x=tool, y=val, text=val, textposition="auto"))