#ifndef SOLVER_H
#define SOLVER_H

#include <petscvec.h>
#include <petscdmda.h>
#include <functional>
#include <string>
#include "flux_scheme.h"

class Mesh;
class MultiBlockMesh;
class FluxDerivative;
class TimeIntegrator;
class SimConfig;
class JacGhostExtentBC;

// ====================================================================
// RHSAssembler — RHS 组装器
//
// 把通量计算 + 度量变换 + 通量求导串起来，完成：
//   RHS = -(∂F̂/∂ξ + ∂Ĝ/∂η + ∂Ĥ/∂ζ) / J
//
// 内部管理通量临时数组 (local Vec)，可跨时间步复用。
// 直接使用 LinearEulerFlux（固定通量方案，无需基类指针）。
// ====================================================================
class RHSAssembler {
public:
    RHSAssembler(const LinearEulerFlux& flux, FluxDerivative* deriv);

    ~RHSAssembler();

    /// @brief 初始化/重建工作数组（当 mesh 变化时调用）
    PetscErrorCode setupWorkVectors(Mesh* mesh);

    /// @brief 释放工作数组
    void cleanupWorkVectors();

    /// @brief 计算右端项
    /// @param mesh   网格块（提供度量系数）
    /// @param gamma  比热比
    /// @param u      扰动守恒变量 5 分量 × 三维（global 索引，ghost 已填好）
    /// @param rhs    输出 RHS 5 分量 × 三维（local 0-based 索引，仅内部格心）
    PetscErrorCode computeRHS(Mesh* mesh, PetscReal gamma,
                              PetscReal*** u[5], PetscReal*** rhs[5]);

private:
    LinearEulerFlux flux_;
    FluxDerivative* deriv_;

    // 缓存的临时 local Vec（每个分量 × 3 方向 + 3 导数）
    Vec Fhat_local_[5], Ghat_local_[5], Hhat_local_[5];
    Vec dFhat_local_[5], dGhat_local_[5], dHhat_local_[5];
    bool work_ready_;
};

// ====================================================================
// 自由函数
// ====================================================================

/// @brief 计算全局时间步长（CFL 条件，线性化 Euler）
/// @param mesh  网格块
/// @param u     扰动守恒变量 U' = [ρ', ρ₀u', ρ₀v', ρ₀w', p']（含 ghost）
/// @param cfl_target  目标 CFL 数
/// @param gamma 比热比
/// @param base_rho  基态密度 ρ₀
/// @param base_p    基态压力 p₀
/// @return dt = cfl_target / max(λ_ξ + λ_η + λ_ζ)
PetscReal computeBlockCFL(Mesh* mesh, PetscReal*** u[5],
                          PetscReal cfl_target, PetscReal gamma,
                          PetscReal base_rho, PetscReal base_p);

/// @brief 计算残差的 L2 范数（用于收敛监控）
/// @param mesh  网格块
/// @param rhs   RHS 5 分量（local 索引）
/// @return sqrt( Σ|rhs|² / N_cells )
PetscReal computeBlockResidual(Mesh* mesh, PetscReal*** rhs[5]);

/// @brief Ghost 填充回调类型
using GhostFillFunc = std::function<PetscErrorCode()>;

/// @brief Euler 求解器主循环（线性化 Euler 方程）
///
/// 遍历所有块，每个时间步：
///   1. 填所有块 ghost
///   2. 算 CFL → dt
///   3. 逐块调用时间积分器
///   4. 输出 + 收敛检查
///
/// @param multiMesh       多块网格
/// @param deriv           导数方案
/// @param time_int        时间积分器
/// @param cfl             CFL 数
/// @param gamma           比热比
/// @param max_steps       最大时间步
/// @param output_interval 输出间隔（步数）
/// @param fillAllGhosts   全局 ghost 填充回调（所有块）
/// @param c0sq            基态声速平方 c₀²
/// @param base_rho        基态密度 ρ₀
/// @param base_p          基态压力 p₀
/// @return 0 成功
PetscErrorCode runEulerSolver(
    MultiBlockMesh& multiMesh,
    FluxDerivative* deriv,
    TimeIntegrator* time_int,
    PetscReal cfl, PetscReal gamma,
    PetscInt max_steps, PetscInt output_interval,
    GhostFillFunc fillAllGhosts,
    PetscReal c0sq, PetscReal base_rho, PetscReal base_p);

/// @brief 求解器一站式入口：注册 BC、构建 ghost 回调、创建组件、运行
///
/// 将 BC 注册、初始 ghost 填充、时间积分器/导数创建、fillGhosts lambda、
/// runEulerSolver 调用封装在一起，main 函数只需调用此接口。
PetscErrorCode setupAndRunSolver(
    MultiBlockMesh& multiMesh,
    const SimConfig& cfg,
    JacGhostExtentBC& ghostFiller);

#endif
