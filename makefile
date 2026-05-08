CXX = mpicxx
CXXFLAGS = -std=c++11 -O2

# PETSc 配置
PETSC_DIR ?= /home/v/PETSc/petsc-3.25.0
PETSC_ARCH ?= arch-linux-c-debug
PETSC_INCLUDE = -I${PETSC_DIR}/include -I${PETSC_DIR}/${PETSC_ARCH}/include
PETSC_LIB = -L${PETSC_DIR}/${PETSC_ARCH}/lib -lpetsc -lm
MPIEXEC = ${PETSC_DIR}/${PETSC_ARCH}/bin/mpiexec

TARGET = main
SRCS = main.cpp mesh_resource.cpp mesh_IO.cpp mesh_jacobi.cpp mesh_ghost.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(PETSC_LIB)

%.o: %.cpp mesh.h
	$(CXX) $(CXXFLAGS) $(PETSC_INCLUDE) -c $<

clean:
	rm -f $(OBJS) $(TARGET)

run: $(TARGET)
	LD_LIBRARY_PATH=${PETSC_DIR}/${PETSC_ARCH}/lib:$$LD_LIBRARY_PATH $(MPIEXEC) -np 4 ./$(TARGET)

.PHONY: all clean run