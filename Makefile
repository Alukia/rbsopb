# install directory of eigen and cplex
EIGENPATH = PATH_TO/eigen_3.2.1
CPLEXPATH = PATH_TO/cplex
CPLEX_LINK = $(CPLEXPATH)/lib/x86-64_linux/static_pic

CXX = g++
# -O3
CXXFLAGS = -g -std=c++11
INCLUDE = -I$(EIGENPATH) -I$(CPLEXPATH)/include
LDFLAGS = -fopenmp -lrt $(CPLEX_LINK)/libcplex.a

SRC = $(wildcard src/*.cpp)
OBJS = $(SRC:src/%.cpp=objs/%.o)
TESTS = $(wildcard tests/*.cpp)
OUT = $(TESTS:tests/%.cpp=bin/%)

.PHONY: all

.SECONDARY: $(OBJS)

all: $(OUT)

readme:
	cat README

bin/%: $(OBJS) tests/%.cpp
	$(CXX) -o $@ $^ $(LDFLAGS) $(INCLUDE)

objs/%.o: src/%.cpp src/%.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) $(INCLUDE)

clean:
	rm $(OUT) $(OBJS)
