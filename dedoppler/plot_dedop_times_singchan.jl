using PlotlyJS

tool = ["turboSETI (CPU)", "turboSETI (GPU)", "SETIcore", "hyperSETI (TBD)"]
val =[24.6, 7.2, 2.0, 0]
plot(bar(x=tool, y=val, text=val, textposition="auto"))