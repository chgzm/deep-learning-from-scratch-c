set grid front
set xlabel "x"
set ylabel "y"
set format y "%.1f"
plot "naive_SGD.txt" w lp t "SGD", "naive_Momentum.txt" w lp t "Momentum", "naive_AdaGrad.txt" w lp t "AdaGrad", "naive_Adam.txt" w lp t "Adam"
pause -1
