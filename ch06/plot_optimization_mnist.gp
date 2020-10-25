set grid front
set xlabel "Iterations"
set ylabel "loss"
set yrange [0:1]
set format y "%.1f"
plot "mnist_SGD.txt" smooth acsplines w l lw 2 t "SGD", "mnist_Momentum.txt" smooth acsplines w l lw 2 t "Momentum", "mnist_AdaGrad.txt" smooth acsplines w l lw 2 t "AdaGrad", "mnist_Adam.txt" smooth acsplines w l lw 2 t "Adam"
#plot "mnist_SGD.txt" w l lw 3p t "SGD", "mnist_Momentum.txt" w l lw 3p t "Momentum", "mnist_AdaGrad.txt" w l lw 3p t "AdaGrad", "mnist_Adam.txt" w l lw 3p t "Adam"
pause -1
