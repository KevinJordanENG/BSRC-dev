using PlotlyJS

tool = ["turboSETI (CPU)", "turboSETI (GPU)", "hyperSETI", "SETIcore"]
val =[24.6, 7.2, 5.9, 2.0]
plot(bar(x=tool, y=val, text=val, textposition="auto"))