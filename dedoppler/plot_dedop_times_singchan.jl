using PlotlyJS

tool = ["turboSETI (CPU)", "turboSETI (GPU)", "hyperSETI", "SETIcore"]
val =[24.6, 7.2, 5.9, 2.0]
plot(bar(x=tool, y=val, text=val, textposition="auto"))

#hyperseti
#t_input_avg = 0.01092950
#t_dedopp_avg = 3.41690416
#t_hits_avg = 2.51472390
#t_out_avg = 0.14411995

