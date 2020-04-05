CFLAGS=-Wall -O3
INCLUDE := -I./../common/
SRCS=$(wildcard ./../common/*.c)
OBJS=$(SRCS:.c=.o)

neuralnet_mnist_batch: $(OBJS)
	gcc neuralnet_mnist_batch.c $(INCLUDE) -o $@ $(OBJS) $(CFLAGS) -lm

clean:
	rm -f neuralnet_mnist_batch ./../common/*.o

.PHONY: clean
