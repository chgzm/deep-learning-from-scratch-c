CC := gcc
CFLAGS := -Wall -O3
INCLUDE := -I./../common/
LIBS := -lm

SRCS := $(wildcard ./../common/*.c)
SRCS += train_neuralnet.c
SRCS += twolayernet.c
OBJS=$(SRCS:.c=.o)

TARGET := train_neuralnet

$(TARGET): $(OBJS)
	$(CC) $(INCLUDE) $(CFLAGS) -o $@ $(OBJS) $(LIBS)

%.o: %.c
	$(CC) $(INCLUDE) $(CFLAGS) -c $< -o $@ $(LIBS)

.PHONY: clean
clean:
	rm -f $(TARGET) $(OBJS)

