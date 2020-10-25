set grid front
set size ratio 0.55
set xlabel "Iterations"
set ylabel "loss"
set yrange [0:1]
set format y "%.1f"
plot "mnist_SGD.txt" smooth acsplines w l t "SGD", "mnist_Momentum.txt" smooth acsplines w l t "Momentum", "mnist_AdaGrad.txt" smooth acsplines w l t "AdaGrad", "mnist_Adam.txt" smooth acsplines w l t "Adam"
pause -1
