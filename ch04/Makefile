CFLAGS=-Wall -O3
INCLUDE := -I./../common/
SRCS=$(wildcard ./../common/*.c)
OBJS=$(SRCS:.c=.o)

all: train_neuralnet

twolayernet.o: twolayernet.c
	gcc -c twolayernet.c $(INCLUDE) $(CFLAGS) -lm

train_neuralnet: $(OBJS) twolayernet.o
	gcc train_neuralnet.c $(INCLUDE) -o $@ $(OBJS) twolayernet.o $(CFLAGS) -lm

clean:
	rm -f train_neuralnet ./../common/*.o ./*.o

.PHONY: clean
