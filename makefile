# PETSc 配置
PETSC_DIR ?= /home/v/PETSc/petsc-3.25.0
PETSC_ARCH ?= arch-linux-c-debug

# 只引入变量（不引入 rules，避免模式和我们的规则冲突）
include $(PETSC_DIR)/lib/petsc/conf/variables

# 编译器：由 PETSc variables 提供 CXX（通常是 mpicxx）
# 编译标志：PETSC_CCPPFLAGS 包含所有 PETSc 头文件路径
CXXFLAGS = -std=c++17 -O2 $(PETSC_CCPPFLAGS)

TARGET = main
SRCS = main.cpp mesh_resource.cpp mesh_IO.cpp mesh_jacobi.cpp mesh_ghost.cpp BC_ghost_filler.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

# 链接：用 CXX 直接链接，加上 PETSc 库
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(PETSC_LIB)

# 编译规则：每个 .cpp 依赖自己的 .h（mesh.h 是公共依赖）
%.o: %.cpp mesh.h mesh_mutiblock.h BC_ghost_filler.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	LD_LIBRARY_PATH=$(PETSC_DIR)/$(PETSC_ARCH)/lib:$$LD_LIBRARY_PATH \
	$(MPIEXEC) -np 16 -bind-to none ./$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET) *.dat

.PHONY: all clean run