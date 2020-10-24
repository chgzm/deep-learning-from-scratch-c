CFLAGS=-Wall -O3
INCLUDE := -I./../common/
SRCS=$(wildcard ./../common/*.c)
OBJS=$(SRCS:.c=.o)
CC=gcc

all: optimizer_compare_naive optimizer_compare_mnist

#optimizer_compare_naive.o: optimizer_compare_naive.c
#	gcc -c optimizer_compare_naive.c $(INCLUDE) $(CFLAGS) -lm

optimizer_compare_naive: $(OBJS) optimizer_compare_naive.c
	gcc optimizer_compare_naive.c $(INCLUDE) -o $@ $(OBJS) $(CFLAGS) -lm

optimizer_compare_mnist: $(OBJS) optimizer_compare_mnist.c
	gcc optimizer_compare_mnist.c $(INCLUDE) -o $@ $(OBJS) $(CFLAGS) -lm


clean:
	rm -f optimizer_compare_naive optimizer_compare_mnist naive_*.txt ./../common/*.o ./*.o

.PHONY: clean
