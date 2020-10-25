set terminal qt size 1800, 900
set grid front
set style fill solid
set multiplot layout 1,5

binwidth=0.03
bin(x,width)=width*floor(x/width)
set size 0.2, 1.0
set title "1-layer"
plot "weight_init_activation_histogram.txt" u (bin($1,binwidth)):(1.0) smooth freq with boxes t ""
set title "2-layer"
set size 0.2, 1.0
plot "weight_init_activation_histogram.txt" u (bin($2,binwidth)):(1.0) smooth freq with boxes t ""
set title "3-layer"
set size 0.2, 1.0
plot "weight_init_activation_histogram.txt" u (bin($3,binwidth)):(1.0) smooth freq with boxes t ""
set title "4-layer"
set size 0.2, 1.0
plot "weight_init_activation_histogram.txt" u (bin($4,binwidth)):(1.0) smooth freq with boxes t ""
set title "5-layer"
set size 0.2, 1.0
plot "weight_init_activation_histogram.txt" u (bin($5,binwidth)):(1.0) smooth freq with boxes t ""

unset multiplot

pause -1
