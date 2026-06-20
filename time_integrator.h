#ifndef TIME_INTEGRATOR_H
#define TIME_INTEGRATOR_H

#include <petscdmda.h>
#include <functional>
#include <string>

class Mesh;
class RHSAssembler;

/// @brief Ghost 填充回调类型
using GhostFillFunc = std::function<PetscErrorCode()>;

// ====================================================================
// TimeIntegrator — 时间推进抽象基类
//
// 给定 U^n 和一个能算 RHS(U) 的 RHSAssembler，
// 按 Butcher 表把 U 推进到 U^{n+1}。
//
// 所有数组均通过 DMDAVecGetArray 获取，使用 global 索引 [k][j][i]。
// ====================================================================
class TimeIntegrator {
public:
    virtual ~TimeIntegrator() = default;

    virtual PetscErrorCode step(
        Mesh* mesh,
        RHSAssembler* rhs_asm,
        PetscReal dt, PetscReal gamma,
        PetscReal*** u[5],
        PetscReal*** rhs[5],
        PetscReal*** u_save[5],
        PetscReal*** u_stage[5],
        GhostFillFunc fillGhosts) = 0;

    // ---- 多块 stage-outer 接口 ----
    // 在 stage 循环前调用一次：保存 U^n、清零累加器
    virtual PetscErrorCode initStageLoop(
        Mesh* mesh,
        PetscReal*** u[5],
        PetscReal*** u_save[5],
        PetscReal*** u_stage[5]) = 0;

    // 每 stage 调用一次：用 computeRHS 的结果按 RK 公式更新 u
    // stage 从 0 开始
    virtual PetscErrorCode applyStage(
        Mesh* mesh, PetscReal dt,
        PetscReal*** u[5], PetscReal*** rhs[5],
        PetscReal*** u_save[5], PetscReal*** u_stage[5],
        int stage) = 0;

    virtual int numStages() const = 0;
    virtual const char* name() const = 0;
};

// ====================================================================
// ForwardEulerIntegrator — 1 阶前向 Euler
// ====================================================================
class ForwardEulerIntegrator : public TimeIntegrator {
public:
    const char* name() const override { return "ForwardEuler"; }
    int numStages() const override { return 1; }
    PetscErrorCode step(Mesh* mesh, RHSAssembler* rhs_asm,
                        PetscReal dt, PetscReal gamma,
                        PetscReal*** u[5], PetscReal*** rhs[5],
                        PetscReal*** u_save[5], PetscReal*** u_stage[5],
                        GhostFillFunc fillGhosts) override;
    PetscErrorCode initStageLoop(Mesh* mesh,
                                  PetscReal*** u[5],
                                  PetscReal*** u_save[5],
                                  PetscReal*** u_stage[5]) override;
    PetscErrorCode applyStage(Mesh* mesh, PetscReal dt,
                               PetscReal*** u[5], PetscReal*** rhs[5],
                               PetscReal*** u_save[5], PetscReal*** u_stage[5],
                               int stage) override;
};

// ====================================================================
// RK3Integrator — 3 阶 TVD Runge-Kutta (Shu-Osher)
//
// U^(1)   = U^n + dt * R(U^n)
// U^(2)   = 3/4 U^n + 1/4 U^(1) + 1/4 dt * R(U^(1))
// U^(n+1) = 1/3 U^n + 2/3 U^(2) + 2/3 dt * R(U^(2))
// ====================================================================
class RK3Integrator : public TimeIntegrator {
public:
    const char* name() const override { return "RK3"; }
    int numStages() const override { return 3; }
    PetscErrorCode step(Mesh* mesh, RHSAssembler* rhs_asm,
                        PetscReal dt, PetscReal gamma,
                        PetscReal*** u[5], PetscReal*** rhs[5],
                        PetscReal*** u_save[5], PetscReal*** u_stage[5],
                        GhostFillFunc fillGhosts) override;
    PetscErrorCode initStageLoop(Mesh* mesh,
                                  PetscReal*** u[5],
                                  PetscReal*** u_save[5],
                                  PetscReal*** u_stage[5]) override;
    PetscErrorCode applyStage(Mesh* mesh, PetscReal dt,
                               PetscReal*** u[5], PetscReal*** rhs[5],
                               PetscReal*** u_save[5], PetscReal*** u_stage[5],
                               int stage) override;
};

// ====================================================================
// RK4Integrator — 4 阶经典 Runge-Kutta
//
// k1 = dt * R(U^n)
// k2 = dt * R(U^n + k1/2)
// k3 = dt * R(U^n + k2/2)
// k4 = dt * R(U^n + k3)
// U^(n+1) = U^n + (k1 + 2*k2 + 2*k3 + k4) / 6
// ====================================================================
class RK4Integrator : public TimeIntegrator {
public:
    const char* name() const override { return "RK4"; }
    int numStages() const override { return 4; }
    PetscErrorCode step(Mesh* mesh, RHSAssembler* rhs_asm,
                        PetscReal dt, PetscReal gamma,
                        PetscReal*** u[5], PetscReal*** rhs[5],
                        PetscReal*** u_save[5], PetscReal*** u_stage[5],
                        GhostFillFunc fillGhosts) override;
    PetscErrorCode initStageLoop(Mesh* mesh,
                                  PetscReal*** u[5],
                                  PetscReal*** u_save[5],
                                  PetscReal*** u_stage[5]) override;
    PetscErrorCode applyStage(Mesh* mesh, PetscReal dt,
                               PetscReal*** u[5], PetscReal*** rhs[5],
                               PetscReal*** u_save[5], PetscReal*** u_stage[5],
                               int stage) override;
};

// ====================================================================
// RK5Integrator — 5 阶段 4 阶 SSP Runge-Kutta (Spiteri-Ruuth)
//
// 系数来自 Spiteri & Ruuth (2002), SSPRK(5,4) 低存储形式：
//   U^(1) = U^n + 0.3917522265718891 * dt * R(U^n)
//   U^(2) = 0.4443704936512352 * U^n + 0.5556295063487648 * U^(1)
//           + 0.3684105930503712 * dt * R(U^(1))
//   U^(3) = 0.6201018514884030 * U^n + 0.3798981485115970 * U^(2)
//           + 0.2518917742716936 * dt * R(U^(2))
//   U^(4) = 0.1780799543931320 * U^n + 0.8219200456068680 * U^(3)
//           + 0.5449747362280225 * dt * R(U^(3))
//   U^(n+1) = 0.5172316719706075 * U^(2) + 0.0960597105261472 * U^(3)
//           + 0.0636924686662900 * dt * R(U^(3))
//           + 0.3867086175032453 * U^(4)
//           + 0.2260074832369060 * dt * R(U^(4))
// ====================================================================
class RK5Integrator : public TimeIntegrator {
public:
    const char* name() const override { return "RK5"; }
    int numStages() const override { return 5; }
    PetscErrorCode step(Mesh* mesh, RHSAssembler* rhs_asm,
                        PetscReal dt, PetscReal gamma,
                        PetscReal*** u[5], PetscReal*** rhs[5],
                        PetscReal*** u_save[5], PetscReal*** u_stage[5],
                        GhostFillFunc fillGhosts) override;
    PetscErrorCode initStageLoop(Mesh* mesh,
                                  PetscReal*** u[5],
                                  PetscReal*** u_save[5],
                                  PetscReal*** u_stage[5]) override;
    PetscErrorCode applyStage(Mesh* mesh, PetscReal dt,
                               PetscReal*** u[5], PetscReal*** rhs[5],
                               PetscReal*** u_save[5], PetscReal*** u_stage[5],
                               int stage) override;
};

// ====================================================================
// 简单工厂函数 — 根据 input.txt 的字符串创建积分器
// ====================================================================
TimeIntegrator* createTimeIntegrator(const std::string& name);

#endif
