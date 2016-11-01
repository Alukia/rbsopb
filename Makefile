# install directory of eigen and cplex
EIGENPATH = /home/nicolas/Applications/eigen_3.2.1
CPLEXPATH = /home/nicolas/Applications/CPLEX/cplex
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
	@mkdir -p bin/
	$(CXX) -o $@ $^ $(LDFLAGS) $(INCLUDE)

objs/%.o: src/%.cpp src/%.h
	@mkdir -p objs/
	$(CXX) -o $@ -c $< $(CXXFLAGS) $(INCLUDE)

clean:
	rm $(OUT) $(OBJS)
