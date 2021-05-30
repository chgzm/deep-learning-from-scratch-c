#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <util.h>
#include <matrix.h>
#include <function.h>
#include "optimizer.h"

static void store_activation(const Matrix* x, double activations[5][100000], int idx) {
    int pos = 0;
    for (int i = 0; i < x->rows; ++i) {
        for (int j = 0; j < x->cols; ++j) {
            activations[idx][pos] = x->elements[i][j];
            ++pos;
        }
    }
}

static void write_activations(double activations[5][100000]) {
    FILE* fp = fopen("weight_init_activation_histogram.txt", "w");
    if (!fp) {
        fprintf(stderr, "Failed to open file.\n");
        return;
    }
    
    for (int i = 0; i < 100000; ++i) {
        fprintf(fp, "%lf %lf %lf %lf %lf\n", activations[0][i], activations[1][i], activations[2][i], activations[3][i], activations[4][i]);
    }

    fclose(fp);
}

int main() {
    Matrix* x = create_matrix(1000, 100);
    init_matrix_random(x);

    const int node_num = 100;
    const int hidden_layer_size = 5;

    double activations[hidden_layer_size][100000];

    for (int i = 0; i < hidden_layer_size; ++i) {
        Matrix* w = create_matrix(node_num, node_num);
        init_matrix_random(w);

        Matrix* a = dot_matrix(x, w);
        free_matrix(x);
        free_matrix(w); 

        x = matrix_sigmoid(a);

        store_activation(x, activations, i);
    }

    write_activations(activations);

    plot_gpfile("plot_weight_init_activation_histogram.gp");

    return 0;   
}
