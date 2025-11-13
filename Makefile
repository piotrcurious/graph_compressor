CXX := g++
CXXFLAGS := -I. -I.. -std=c++11 -Wall -Wextra
LDFLAGS :=

# Add Eigen library path directly.
CXXFLAGS += -I/usr/include/eigen3


SRCS := main.cpp multi_poly6a2_total_tidy_algebraic2_tidy3_lowmem3_newgraph2/AdvancedPolynomialFitter.cpp
OBJS := $(SRCS:.cpp=.o)
TARGET := run_test

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Create a mock Arduino header to allow compilation on Linux
Arduino.h:
	@echo "Creating mock Arduino.h..."
	@echo "// Mock Arduino.h for non-Arduino compilation" > Arduino.h
	@echo "#ifndef ARDUINO_H" >> Arduino.h
	@echo "#define ARDUINO_H" >> Arduino.h
	@echo "#include <cstdint>" >> Arduino.h
	@echo "#include <cmath>" >> Arduino.h
	@echo "#define ARDUINO 101 // Mock version" >> Arduino.h
	@echo "class MockSerial { public: int available() { return 0; } void printf(...) {} };" >> Arduino.h
	@echo "extern MockSerial Serial;" >> Arduino.h
	@echo "#endif" >> Arduino.h

# Ensure mock header is created before compiling
main.o: Arduino.h
multi_poly6a2_total_tidy_algebraic2_tidy3_lowmem3_newgraph2/AdvancedPolynomialFitter.o: Arduino.h

clean:
	rm -f $(OBJS) $(TARGET) Arduino.h
