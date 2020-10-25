set grid front
set xlabel "Iterations"
set ylabel "loss"
#set yrange [0:2.5]
set format y "%.1f"
plot "weight_init_compare_He.txt" smooth acsplines w l lw 2 t "He", "weight_init_compare_STD.txt" smooth acsplines w l lw 2 t "std=0.01", "weight_init_compare_Xavier.txt" smooth acsplines w l lw 2 t "Xavier"
pause -1
