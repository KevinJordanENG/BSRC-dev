using PlotlyJS

tool = ["turboSETI (CPU)", "turboSETI (GPU)", "SETIcore", "hyperSETI (TBD)"]
val =[22.1, 3.31, 0.9, 0]
plot(bar(x=tool, y=val, text=val, textposition="auto"))