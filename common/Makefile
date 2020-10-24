CFLAGS=-Wall -O3
INCLUDE := -I./../common/
SRCS=$(wildcard ./../common/*.c)
OBJS=$(SRCS:.c=.o)
CC=gcc

all: $(OBJS)

clean:
	rm -f *.o

.PHONY: clean
