CC := gcc
CFLAGS := -Wall -O3
INCLUDE := -I./../common/
LIBS := -lm

SRCS := $(wildcard ./../common/*.c)
OBJS := $(SRCS:.c=.o)

TARGETS := optimizer_compare_naive optimizer_compare_mnist weight_init_activation_histogram weight_init_compare batch_norm_test overfit_weight_decay overfit_dropout hyperparameter_optimization

all: $(TARGETS) 

optimizer_compare_naive: optimizer_compare_naive.c $(OBJS)
	$(CC) $(INCLUDE) $(CFLAGS) -o $@ $< $(OBJS) $(LIBS)

optimizer_compare_mnist: optimizer_compare_mnist.c $(OBJS)
	$(CC) $(INCLUDE) $(CFLAGS) -o $@ $< $(OBJS) $(LIBS)

weight_init_activation_histogram: weight_init_activation_histogram.c $(OBJS)
	$(CC) $(INCLUDE) $(CFLAGS) -o $@ $< $(OBJS) $(LIBS)

weight_init_compare: weight_init_compare.c $(OBJS)
	$(CC) $(INCLUDE) $(CFLAGS) -o $@ $< $(OBJS) $(LIBS)

batch_norm_test: batch_norm_test.c $(OBJS)
	$(CC) $(INCLUDE) $(CFLAGS) -o $@ $< $(OBJS) $(LIBS)

overfit_weight_decay: overfit_weight_decay.c $(OBJS)
	$(CC) $(INCLUDE) $(CFLAGS) -o $@ $< $(OBJS) $(LIBS)

overfit_dropout: overfit_dropout.c $(OBJS)
	$(CC) $(INCLUDE) $(CFLAGS) -o $@ $< $(OBJS) $(LIBS)

hyperparameter_optimization: hyperparameter_optimization.c $(OBJS)
	$(CC) $(INCLUDE) $(CFLAGS) -o $@ $< $(OBJS) $(LIBS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(OBJS)

