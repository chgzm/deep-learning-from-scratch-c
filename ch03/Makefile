CC := gcc
CFLAGS := -Wall -O3
INCLUDE := -I./../common/

SRCS := $(wildcard ./../common/*.c)
SRCS += neuralnet_mnist_batch.c
OBJS := $(SRCS:.c=.o)
LIBS := -lm
TARGET := neuralnet_mnist_batch

$(TARGET): $(OBJS)
	$(CC) $(INCLUDE) -o $@ $(OBJS) $(CFLAGS) $(LIBS)

%.o: %.c
	$(CC) $(INCLUDE) $(CFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f $(TARGET) $(OBJS)

