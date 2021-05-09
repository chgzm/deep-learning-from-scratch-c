CC := gcc
CFLAGS := -Wall -O3

SRCS := $(wildcard ./../common/*.c)
OBJS := $(SRCS:.c=.o)

all: $(OBJS)

.PHONY: clean
clean:
	rm -f *.o
