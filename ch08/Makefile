CC := gcc
CFLAGS := -Wall -O3 -g
INCLUDE := -I./../common/
LIBS := -lm

SRCS := $(wildcard ./../common/*.c)
SRCS += deep_convnet.c
OBJS := $(SRCS:.c=.o)

TARGETS := misclassified_mnist

all: $(TARGETS)

misclassified_mnist: misclassified_mnist.c $(OBJS) 
	$(CC) $(INCLUDE) $(CFLAGS) -o $@ $< $(OBJS) $(LIBS)

%.o: %.c
	$(CC) $(INCLUDE) $(CFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f $(TARGETS) $(OBJS)

