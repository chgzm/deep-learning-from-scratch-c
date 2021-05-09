CC := gcc
CXX := g++
CFLAGS := -Wall -O3
CXXFLAGS := -Wall -O3 -std=c++14
INCLUDE := -I../../common
LIB := -lgtest -lgtest_main -lpthread

CSRCS := $(wildcard ../../common/*.c)
CXXSRCS := $(wildcard *.cpp)
SRCS := $(CSRCS) $(CXXSRCS)
OBJS := $(CSRCS:%.c=%.o) $(CXXSRCS:%.cpp=%.o)

TARGET := gtest

$(TARGET): $(OBJS)
	$(CXX) $(INCLUDE) $(CXXFLAGS) -o $@ $(OBJS) $(LIB)

%.o: %.cpp
	$(CXX) $(INCLUDE) $(CXXFLAGS) -c $< -o $@ $(LIBS)

%.o: %.c
	$(CC) $(INCLUDE) $(CFLAGS) -c $< -o $@ $(LIBS)

.PHONY: clean
clean:
	rm -f $(TARGET) $(OBJS)
