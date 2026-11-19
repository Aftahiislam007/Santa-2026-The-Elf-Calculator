CXX ?= clang++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra

SOURCES := $(wildcard *.cpp)
BINARIES := $(patsubst %.cpp,bin/%,$(SOURCES))

all: $(BINARIES)

bin/%: %.cpp $(wildcard *.hpp)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -rf bin

.PHONY: all clean
