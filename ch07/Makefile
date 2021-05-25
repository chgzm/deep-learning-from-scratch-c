CC := gcc
CFLAGS := -Wall -O3 -g
INCLUDE := -I./../common/
LIBS := -lm

SRCS := $(wildcard ./../common/*.c)
OBJS := $(SRCS:.c=.o)

TARGETS := train_convnet

all: $(TARGETS)

train_convnet: train_convnet.c $(OBJS) 
	$(CC) $(INCLUDE) $(CFLAGS) -o $@ $< $(OBJS) $(LIBS)

%.o: %.c
	$(CC) $(INCLUDE) $(CFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f $(TARGETS) $(OBJS)

