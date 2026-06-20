# PETSc 配置
PETSC_DIR ?= /home/v/PETSc/petsc-3.25.0
PETSC_ARCH ?= arch-linux-c-debug

include $(PETSC_DIR)/lib/petsc/conf/variables

# CGNS 配置（根据你的实际路径修改）
CGNS_DIR ?= /usr/local/cgns
CGNS_LIB  = -L$(CGNS_DIR)/lib -lcgns
CGNS_INC  = -I$(CGNS_DIR)/include

CXXFLAGS = -std=c++17 -O2 $(PETSC_CCPPFLAGS) $(CGNS_INC)

TARGET = main
SRCS = main.cpp mesh_resource.cpp mesh_IO.cpp mesh_jacobi.cpp \
       mesh_ghost.cpp BC_ghost_filler.cpp BC_filler.cpp sim_config.cpp flow_init.cpp \
       flux_derivative.cpp solver.cpp time_integrator.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(PETSC_LIB) $(CGNS_LIB)

%.o: %.cpp mesh.h mesh_mutiblock.h BC_ghost_filler.h BC_filler.h sim_config.h flow_init.h \
       flux_scheme.h flux_derivative.h solver.h time_integrator.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	LD_LIBRARY_PATH=$(PETSC_DIR)/$(PETSC_ARCH)/lib:$$LD_LIBRARY_PATH \
	$(MPIEXEC) -np 16 -bind-to none ./$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET) *.dat

.PHONY: all clean run
