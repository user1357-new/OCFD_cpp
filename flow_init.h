#ifndef FLOW_INIT_H
#define FLOW_INIT_H

#include <petsc.h>

class MultiBlockMesh;
class SimConfig;

/// @brief 统一的流场初始化入口（替代原来的工厂模式）
PetscErrorCode initializeFlowField(MultiBlockMesh& mesh, const SimConfig& cfg);

/// @brief 写出二进制重启文件（格心守恒量），格式与 datInit 读取匹配
PetscErrorCode writeRestart(MultiBlockMesh& mesh, const std::string& filename);

#endif
