#include <stdio.h>
#include <matrix.h>
#include <simple_convnet.h>

static void filter_show(const Matrix4d* W, const char* title) {
    FILE* gp = popen("gnuplot -persist", "w");
    if (gp == NULL) {
        fprintf(stderr, "Failed to open gnuplot pipe.\n");
        return;
    }

    fprintf(gp, "set term qt size 1000, 500\n");
    fprintf(gp, "set palette defined (0 'white', 1 'black')\n");
    fprintf(gp, "unset colorbox\n");
    fprintf(gp, "unset key\n");
    fprintf(gp, "unset tics\n");
    fprintf(gp, "set multiplot layout 4, 8 title '%s'\n", title);
    fprintf(gp, "set yrange [] reverse\n");
    for (int i = 0; i < W->sizes[0]; ++i) {
        fprintf(gp, "plot '-' matrix with image\n");
        for (int j = 0; j < W->sizes[2]; ++j) {
            for (int k = 0; k < W->sizes[3]; ++k) {
                fprintf(gp, "%lf ", W->elements[i][0][j][k]);
            }
            fprintf(gp, "\n");
        }
        fprintf(gp, "e\n");
        fprintf(gp, "e\n");
        fflush(gp);
    }

    fclose(gp);
}

int main() {
    SimpleConvNet* net = create_simple_convnet(1, 28, 28, 30, 5, 0, 1, 100, 10, 0.01);

    filter_show(net->C->W, "Initialized");
    simple_convnet_load_params(net);
    filter_show(net->C->W, "Trained");

    return 0;   
}
