#include "time_integrator.h"
#include "solver.h"
#include "mesh.h"
#include <petsc.h>

// ====================================================================
// 辅助：获取 DMDA local info
// ====================================================================
static void getLocalRange(DM da,
                           PetscInt &xs, PetscInt &ys, PetscInt &zs,
                           PetscInt &xm, PetscInt &ym, PetscInt &zm)
{
    DMDALocalInfo info;
    DMDAGetLocalInfo(da, &info);
    xs = info.xs; xm = info.xm;
    ys = info.ys; ym = info.ym;
    zs = info.zs; zm = info.zm;
}

// ====================================================================
// ForwardEulerIntegrator
// ====================================================================
PetscErrorCode ForwardEulerIntegrator::step(
    Mesh* mesh, RHSAssembler* rhs_asm,
    PetscReal dt, PetscReal gamma,
    PetscReal*** u[5], PetscReal*** rhs[5],
    PetscReal*** u_save[5], PetscReal*** u_stage[5],
    GhostFillFunc fillGhosts)
{
    PetscErrorCode ierr;
    DM da = mesh->getDM();

    PetscInt xs, ys, zs, xm, ym, zm;
    getLocalRange(da, xs, ys, zs, xm, ym, zm);

    // RHS(U^n)
    ierr = fillGhosts(); CHKERRQ(ierr);
    ierr = rhs_asm->computeRHS(mesh, gamma, u, rhs); CHKERRQ(ierr);

    // U^{n+1} = U^n + dt * RHS
    for (PetscInt k = zs; k < zs + zm; k++) {
        for (PetscInt j = ys; j < ys + ym; j++) {
            for (PetscInt i = xs; i < xs + xm; i++) {
                u[0][k][j][i] += dt * rhs[0][k][j][i];
                u[1][k][j][i] += dt * rhs[1][k][j][i];
                u[2][k][j][i] += dt * rhs[2][k][j][i];
                u[3][k][j][i] += dt * rhs[3][k][j][i];
                u[4][k][j][i] += dt * rhs[4][k][j][i];
            }
        }
    }

    return 0;
}

PetscErrorCode ForwardEulerIntegrator::initStageLoop(
    Mesh* mesh, PetscReal*** u[5],
    PetscReal*** u_save[5], PetscReal*** u_stage[5])
{
    // Forward Euler 单 stage，无需保存状态
    (void)mesh; (void)u; (void)u_save; (void)u_stage;
    return 0;
}

PetscErrorCode ForwardEulerIntegrator::applyStage(
    Mesh* mesh, PetscReal dt,
    PetscReal*** u[5], PetscReal*** rhs[5],
    PetscReal*** u_save[5], PetscReal*** u_stage[5],
    int stage)
{
    DM da = mesh->getDM();
    PetscInt xs, ys, zs, xm, ym, zm;
    getLocalRange(da, xs, ys, zs, xm, ym, zm);

    (void)u_save; (void)u_stage;

    // U^{n+1} = U^n + dt * RHS
    for (PetscInt k = zs; k < zs + zm; k++) {
        for (PetscInt j = ys; j < ys + ym; j++) {
            for (PetscInt i = xs; i < xs + xm; i++) {
                u[0][k][j][i] += dt * rhs[0][k][j][i];
                u[1][k][j][i] += dt * rhs[1][k][j][i];
                u[2][k][j][i] += dt * rhs[2][k][j][i];
                u[3][k][j][i] += dt * rhs[3][k][j][i];
                u[4][k][j][i] += dt * rhs[4][k][j][i];
            }
        }
    }

    (void)stage;
    return 0;
}

// ====================================================================
// RK3Integrator — TVD RK3 (Shu-Osher)
//
// U^(1)   = U^n + dt * R(U^n)
// U^(2)   = 3/4 U^n + 1/4 U^(1) + 1/4 dt * R(U^(1))
// U^(n+1) = 1/3 U^n + 2/3 U^(2) + 2/3 dt * R(U^(2))
// ====================================================================
PetscErrorCode RK3Integrator::step(
    Mesh* mesh, RHSAssembler* rhs_asm,
    PetscReal dt, PetscReal gamma,
    PetscReal*** u[5], PetscReal*** rhs[5],
    PetscReal*** u_save[5], PetscReal*** u_stage[5],
    GhostFillFunc fillGhosts)
{
    PetscErrorCode ierr;
    DM da = mesh->getDM();

    PetscInt xs, ys, zs, xm, ym, zm;
    getLocalRange(da, xs, ys, zs, xm, ym, zm);

    // ====== Stage 1: k1 = R(U^n) ======
    ierr = fillGhosts(); CHKERRQ(ierr);
    ierr = rhs_asm->computeRHS(mesh, gamma, u, rhs); CHKERRQ(ierr);

    // u_save = U^n,  u = U^n + dt*k1
    for (PetscInt k = zs; k < zs + zm; k++) {
        for (PetscInt j = ys; j < ys + ym; j++) {
            for (PetscInt i = xs; i < xs + xm; i++) {
                u_save[0][k][j][i] = u[0][k][j][i];
                u_save[1][k][j][i] = u[1][k][j][i];
                u_save[2][k][j][i] = u[2][k][j][i];
                u_save[3][k][j][i] = u[3][k][j][i];
                u_save[4][k][j][i] = u[4][k][j][i];

                u[0][k][j][i] += dt * rhs[0][k][j][i];
                u[1][k][j][i] += dt * rhs[1][k][j][i];
                u[2][k][j][i] += dt * rhs[2][k][j][i];
                u[3][k][j][i] += dt * rhs[3][k][j][i];
                u[4][k][j][i] += dt * rhs[4][k][j][i];
            }
        }
    }

    // ====== Stage 2: k2 = R(U^(1)), U^(2) = 3/4 U^n + 1/4 U^(1) + 1/4 dt*k2 ======
    ierr = fillGhosts(); CHKERRQ(ierr);
    ierr = rhs_asm->computeRHS(mesh, gamma, u, rhs); CHKERRQ(ierr);

    for (PetscInt k = zs; k < zs + zm; k++) {
        for (PetscInt j = ys; j < ys + ym; j++) {
            for (PetscInt i = xs; i < xs + xm; i++) {
                u[0][k][j][i] = 0.75 * u_save[0][k][j][i]
                              + 0.25 * u[0][k][j][i]       // U^(1)
                              + 0.25 * dt * rhs[0][k][j][i];
                u[1][k][j][i] = 0.75 * u_save[1][k][j][i]
                              + 0.25 * u[1][k][j][i]
                              + 0.25 * dt * rhs[1][k][j][i];
                u[2][k][j][i] = 0.75 * u_save[2][k][j][i]
                              + 0.25 * u[2][k][j][i]
                              + 0.25 * dt * rhs[2][k][j][i];
                u[3][k][j][i] = 0.75 * u_save[3][k][j][i]
                              + 0.25 * u[3][k][j][i]
                              + 0.25 * dt * rhs[3][k][j][i];
                u[4][k][j][i] = 0.75 * u_save[4][k][j][i]
                              + 0.25 * u[4][k][j][i]
                              + 0.25 * dt * rhs[4][k][j][i];
            }
        }
    }

    // ====== Stage 3: k3 = R(U^(2)), U^(n+1) = 1/3 U^n + 2/3 U^(2) + 2/3 dt*k3 ======
    ierr = fillGhosts(); CHKERRQ(ierr);
    ierr = rhs_asm->computeRHS(mesh, gamma, u, rhs); CHKERRQ(ierr);

    for (PetscInt k = zs; k < zs + zm; k++) {
        for (PetscInt j = ys; j < ys + ym; j++) {
            for (PetscInt i = xs; i < xs + xm; i++) {
                u[0][k][j][i] = (1.0/3.0) * u_save[0][k][j][i]
                              + (2.0/3.0) * u[0][k][j][i]       // U^(2)
                              + (2.0/3.0) * dt * rhs[0][k][j][i];
                u[1][k][j][i] = (1.0/3.0) * u_save[1][k][j][i]
                              + (2.0/3.0) * u[1][k][j][i]
                              + (2.0/3.0) * dt * rhs[1][k][j][i];
                u[2][k][j][i] = (1.0/3.0) * u_save[2][k][j][i]
                              + (2.0/3.0) * u[2][k][j][i]
                              + (2.0/3.0) * dt * rhs[2][k][j][i];
                u[3][k][j][i] = (1.0/3.0) * u_save[3][k][j][i]
                              + (2.0/3.0) * u[3][k][j][i]
                              + (2.0/3.0) * dt * rhs[3][k][j][i];
                u[4][k][j][i] = (1.0/3.0) * u_save[4][k][j][i]
                              + (2.0/3.0) * u[4][k][j][i]
                              + (2.0/3.0) * dt * rhs[4][k][j][i];
            }
        }
    }

    return 0;
}

PetscErrorCode RK3Integrator::initStageLoop(
    Mesh* mesh, PetscReal*** u[5],
    PetscReal*** u_save[5], PetscReal*** u_stage[5])
{
    DM da = mesh->getDM();
    PetscInt xs, ys, zs, xm, ym, zm;
    getLocalRange(da, xs, ys, zs, xm, ym, zm);

    (void)u_stage;

    // 保存 U^n → u_save
    for (PetscInt k = zs; k < zs + zm; k++) {
        for (PetscInt j = ys; j < ys + ym; j++) {
            for (PetscInt i = xs; i < xs + xm; i++) {
                u_save[0][k][j][i] = u[0][k][j][i];
                u_save[1][k][j][i] = u[1][k][j][i];
                u_save[2][k][j][i] = u[2][k][j][i];
                u_save[3][k][j][i] = u[3][k][j][i];
                u_save[4][k][j][i] = u[4][k][j][i];
            }
        }
    }
    return 0;
}

PetscErrorCode RK3Integrator::applyStage(
    Mesh* mesh, PetscReal dt,
    PetscReal*** u[5], PetscReal*** rhs[5],
    PetscReal*** u_save[5], PetscReal*** u_stage[5],
    int stage)
{
    DM da = mesh->getDM();
    PetscInt xs, ys, zs, xm, ym, zm;
    getLocalRange(da, xs, ys, zs, xm, ym, zm);

    (void)u_stage;

    switch (stage) {
    case 0:  // U^(1) = U^n + dt * R(U^n)
        for (PetscInt k = zs; k < zs + zm; k++) {
            for (PetscInt j = ys; j < ys + ym; j++) {
                for (PetscInt i = xs; i < xs + xm; i++) {
                    u[0][k][j][i] = u_save[0][k][j][i] + dt * rhs[0][k][j][i];
                    u[1][k][j][i] = u_save[1][k][j][i] + dt * rhs[1][k][j][i];
                    u[2][k][j][i] = u_save[2][k][j][i] + dt * rhs[2][k][j][i];
                    u[3][k][j][i] = u_save[3][k][j][i] + dt * rhs[3][k][j][i];
                    u[4][k][j][i] = u_save[4][k][j][i] + dt * rhs[4][k][j][i];
                }
            }
        }
        break;
    case 1:  // U^(2) = 3/4 U^n + 1/4 U^(1) + 1/4 dt * R(U^(1))
        for (PetscInt k = zs; k < zs + zm; k++) {
            for (PetscInt j = ys; j < ys + ym; j++) {
                for (PetscInt i = xs; i < xs + xm; i++) {
                    u[0][k][j][i] = 0.75 * u_save[0][k][j][i]
                                  + 0.25 * u[0][k][j][i]
                                  + 0.25 * dt * rhs[0][k][j][i];
                    u[1][k][j][i] = 0.75 * u_save[1][k][j][i]
                                  + 0.25 * u[1][k][j][i]
                                  + 0.25 * dt * rhs[1][k][j][i];
                    u[2][k][j][i] = 0.75 * u_save[2][k][j][i]
                                  + 0.25 * u[2][k][j][i]
                                  + 0.25 * dt * rhs[2][k][j][i];
                    u[3][k][j][i] = 0.75 * u_save[3][k][j][i]
                                  + 0.25 * u[3][k][j][i]
                                  + 0.25 * dt * rhs[3][k][j][i];
                    u[4][k][j][i] = 0.75 * u_save[4][k][j][i]
                                  + 0.25 * u[4][k][j][i]
                                  + 0.25 * dt * rhs[4][k][j][i];
                }
            }
        }
        break;
    case 2:  // U^(n+1) = 1/3 U^n + 2/3 U^(2) + 2/3 dt * R(U^(2))
        for (PetscInt k = zs; k < zs + zm; k++) {
            for (PetscInt j = ys; j < ys + ym; j++) {
                for (PetscInt i = xs; i < xs + xm; i++) {
                    u[0][k][j][i] = (1.0/3.0) * u_save[0][k][j][i]
                                  + (2.0/3.0) * u[0][k][j][i]
                                  + (2.0/3.0) * dt * rhs[0][k][j][i];
                    u[1][k][j][i] = (1.0/3.0) * u_save[1][k][j][i]
                                  + (2.0/3.0) * u[1][k][j][i]
                                  + (2.0/3.0) * dt * rhs[1][k][j][i];
                    u[2][k][j][i] = (1.0/3.0) * u_save[2][k][j][i]
                                  + (2.0/3.0) * u[2][k][j][i]
                                  + (2.0/3.0) * dt * rhs[2][k][j][i];
                    u[3][k][j][i] = (1.0/3.0) * u_save[3][k][j][i]
                                  + (2.0/3.0) * u[3][k][j][i]
                                  + (2.0/3.0) * dt * rhs[3][k][j][i];
                    u[4][k][j][i] = (1.0/3.0) * u_save[4][k][j][i]
                                  + (2.0/3.0) * u[4][k][j][i]
                                  + (2.0/3.0) * dt * rhs[4][k][j][i];
                }
            }
        }
        break;
    }
    return 0;
}

// ====================================================================
// RK4Integrator — 经典 4 阶 RK
//
// k1 = dt * R(U^n)
// k2 = dt * R(U^n + k1/2)
// k3 = dt * R(U^n + k2/2)
// k4 = dt * R(U^n + k3)
// U^(n+1) = U^n + (k1 + 2*k2 + 2*k3 + k4) / 6
//
// u_stage 用作累加器：Σ = k1 + 2*k2 + 2*k3 + k4
// ====================================================================
PetscErrorCode RK4Integrator::step(
    Mesh* mesh, RHSAssembler* rhs_asm,
    PetscReal dt, PetscReal gamma,
    PetscReal*** u[5], PetscReal*** rhs[5],
    PetscReal*** u_save[5], PetscReal*** u_stage[5],
    GhostFillFunc fillGhosts)
{
    PetscErrorCode ierr;
    DM da = mesh->getDM();

    PetscInt xs, ys, zs, xm, ym, zm;
    getLocalRange(da, xs, ys, zs, xm, ym, zm);

    PetscReal dt_half = 0.5 * dt;

    // ====== Stage 1: k1 = dt * R(U^n) ======
    ierr = fillGhosts(); CHKERRQ(ierr);
    ierr = rhs_asm->computeRHS(mesh, gamma, u, rhs); CHKERRQ(ierr);

    for (PetscInt k = zs; k < zs + zm; k++) {
        for (PetscInt j = ys; j < ys + ym; j++) {
            for (PetscInt i = xs; i < xs + xm; i++) {
                // 保存 U^n
                u_save[0][k][j][i] = u[0][k][j][i];
                u_save[1][k][j][i] = u[1][k][j][i];
                u_save[2][k][j][i] = u[2][k][j][i];
                u_save[3][k][j][i] = u[3][k][j][i];
                u_save[4][k][j][i] = u[4][k][j][i];

                // 累加器 = k1
                u_stage[0][k][j][i] = dt * rhs[0][k][j][i];
                u_stage[1][k][j][i] = dt * rhs[1][k][j][i];
                u_stage[2][k][j][i] = dt * rhs[2][k][j][i];
                u_stage[3][k][j][i] = dt * rhs[3][k][j][i];
                u_stage[4][k][j][i] = dt * rhs[4][k][j][i];

                // U^(1) = U^n + k1/2
                u[0][k][j][i] = u_save[0][k][j][i] + dt_half * rhs[0][k][j][i];
                u[1][k][j][i] = u_save[1][k][j][i] + dt_half * rhs[1][k][j][i];
                u[2][k][j][i] = u_save[2][k][j][i] + dt_half * rhs[2][k][j][i];
                u[3][k][j][i] = u_save[3][k][j][i] + dt_half * rhs[3][k][j][i];
                u[4][k][j][i] = u_save[4][k][j][i] + dt_half * rhs[4][k][j][i];
            }
        }
    }

    // ====== Stage 2: k2 = dt * R(U^(1)) ======
    ierr = fillGhosts(); CHKERRQ(ierr);
    ierr = rhs_asm->computeRHS(mesh, gamma, u, rhs); CHKERRQ(ierr);

    for (PetscInt k = zs; k < zs + zm; k++) {
        for (PetscInt j = ys; j < ys + ym; j++) {
            for (PetscInt i = xs; i < xs + xm; i++) {
                // 累加器 += 2*k2
                u_stage[0][k][j][i] += 2.0 * dt * rhs[0][k][j][i];
                u_stage[1][k][j][i] += 2.0 * dt * rhs[1][k][j][i];
                u_stage[2][k][j][i] += 2.0 * dt * rhs[2][k][j][i];
                u_stage[3][k][j][i] += 2.0 * dt * rhs[3][k][j][i];
                u_stage[4][k][j][i] += 2.0 * dt * rhs[4][k][j][i];

                // U^(2) = U^n + k2/2
                u[0][k][j][i] = u_save[0][k][j][i] + dt_half * rhs[0][k][j][i];
                u[1][k][j][i] = u_save[1][k][j][i] + dt_half * rhs[1][k][j][i];
                u[2][k][j][i] = u_save[2][k][j][i] + dt_half * rhs[2][k][j][i];
                u[3][k][j][i] = u_save[3][k][j][i] + dt_half * rhs[3][k][j][i];
                u[4][k][j][i] = u_save[4][k][j][i] + dt_half * rhs[4][k][j][i];
            }
        }
    }

    // ====== Stage 3: k3 = dt * R(U^(2)) ======
    ierr = fillGhosts(); CHKERRQ(ierr);
    ierr = rhs_asm->computeRHS(mesh, gamma, u, rhs); CHKERRQ(ierr);

    for (PetscInt k = zs; k < zs + zm; k++) {
        for (PetscInt j = ys; j < ys + ym; j++) {
            for (PetscInt i = xs; i < xs + xm; i++) {
                // 累加器 += 2*k3
                u_stage[0][k][j][i] += 2.0 * dt * rhs[0][k][j][i];
                u_stage[1][k][j][i] += 2.0 * dt * rhs[1][k][j][i];
                u_stage[2][k][j][i] += 2.0 * dt * rhs[2][k][j][i];
                u_stage[3][k][j][i] += 2.0 * dt * rhs[3][k][j][i];
                u_stage[4][k][j][i] += 2.0 * dt * rhs[4][k][j][i];

                // U^(3) = U^n + k3
                u[0][k][j][i] = u_save[0][k][j][i] + dt * rhs[0][k][j][i];
                u[1][k][j][i] = u_save[1][k][j][i] + dt * rhs[1][k][j][i];
                u[2][k][j][i] = u_save[2][k][j][i] + dt * rhs[2][k][j][i];
                u[3][k][j][i] = u_save[3][k][j][i] + dt * rhs[3][k][j][i];
                u[4][k][j][i] = u_save[4][k][j][i] + dt * rhs[4][k][j][i];
            }
        }
    }

    // ====== Stage 4: k4 = dt * R(U^(3)) ======
    ierr = fillGhosts(); CHKERRQ(ierr);
    ierr = rhs_asm->computeRHS(mesh, gamma, u, rhs); CHKERRQ(ierr);

    PetscReal inv6 = 1.0 / 6.0;

    for (PetscInt k = zs; k < zs + zm; k++) {
        for (PetscInt j = ys; j < ys + ym; j++) {
            for (PetscInt i = xs; i < xs + xm; i++) {
                // 累加器 += k4
                u_stage[0][k][j][i] += dt * rhs[0][k][j][i];
                u_stage[1][k][j][i] += dt * rhs[1][k][j][i];
                u_stage[2][k][j][i] += dt * rhs[2][k][j][i];
                u_stage[3][k][j][i] += dt * rhs[3][k][j][i];
                u_stage[4][k][j][i] += dt * rhs[4][k][j][i];

                // 最终：U^(n+1) = U^n + (k1 + 2*k2 + 2*k3 + k4) / 6
                u[0][k][j][i] = u_save[0][k][j][i] + u_stage[0][k][j][i] * inv6;
                u[1][k][j][i] = u_save[1][k][j][i] + u_stage[1][k][j][i] * inv6;
                u[2][k][j][i] = u_save[2][k][j][i] + u_stage[2][k][j][i] * inv6;
                u[3][k][j][i] = u_save[3][k][j][i] + u_stage[3][k][j][i] * inv6;
                u[4][k][j][i] = u_save[4][k][j][i] + u_stage[4][k][j][i] * inv6;
            }
        }
    }

    return 0;
}

PetscErrorCode RK4Integrator::initStageLoop(
    Mesh* mesh, PetscReal*** u[5],
    PetscReal*** u_save[5], PetscReal*** u_stage[5])
{
    DM da = mesh->getDM();
    PetscInt xs, ys, zs, xm, ym, zm;
    getLocalRange(da, xs, ys, zs, xm, ym, zm);

    // 保存 U^n → u_save，清零累加器 u_stage
    for (PetscInt k = zs; k < zs + zm; k++) {
        for (PetscInt j = ys; j < ys + ym; j++) {
            for (PetscInt i = xs; i < xs + xm; i++) {
                u_save[0][k][j][i] = u[0][k][j][i];
                u_save[1][k][j][i] = u[1][k][j][i];
                u_save[2][k][j][i] = u[2][k][j][i];
                u_save[3][k][j][i] = u[3][k][j][i];
                u_save[4][k][j][i] = u[4][k][j][i];

                u_stage[0][k][j][i] = 0.0;
                u_stage[1][k][j][i] = 0.0;
                u_stage[2][k][j][i] = 0.0;
                u_stage[3][k][j][i] = 0.0;
                u_stage[4][k][j][i] = 0.0;
            }
        }
    }
    return 0;
}

PetscErrorCode RK4Integrator::applyStage(
    Mesh* mesh, PetscReal dt,
    PetscReal*** u[5], PetscReal*** rhs[5],
    PetscReal*** u_save[5], PetscReal*** u_stage[5],
    int stage)
{
    DM da = mesh->getDM();
    PetscInt xs, ys, zs, xm, ym, zm;
    getLocalRange(da, xs, ys, zs, xm, ym, zm);

    switch (stage) {
    case 0:  // k1: acc = dt*RHS, u = U^n + k1/2
        for (PetscInt k = zs; k < zs + zm; k++) {
            for (PetscInt j = ys; j < ys + ym; j++) {
                for (PetscInt i = xs; i < xs + xm; i++) {
                    u_stage[0][k][j][i] = dt * rhs[0][k][j][i];
                    u_stage[1][k][j][i] = dt * rhs[1][k][j][i];
                    u_stage[2][k][j][i] = dt * rhs[2][k][j][i];
                    u_stage[3][k][j][i] = dt * rhs[3][k][j][i];
                    u_stage[4][k][j][i] = dt * rhs[4][k][j][i];

                    u[0][k][j][i] = u_save[0][k][j][i] + 0.5 * dt * rhs[0][k][j][i];
                    u[1][k][j][i] = u_save[1][k][j][i] + 0.5 * dt * rhs[1][k][j][i];
                    u[2][k][j][i] = u_save[2][k][j][i] + 0.5 * dt * rhs[2][k][j][i];
                    u[3][k][j][i] = u_save[3][k][j][i] + 0.5 * dt * rhs[3][k][j][i];
                    u[4][k][j][i] = u_save[4][k][j][i] + 0.5 * dt * rhs[4][k][j][i];
                }
            }
        }
        break;
    case 1:  // k2: acc += 2*k2, u = U^n + k2/2
        for (PetscInt k = zs; k < zs + zm; k++) {
            for (PetscInt j = ys; j < ys + ym; j++) {
                for (PetscInt i = xs; i < xs + xm; i++) {
                    u_stage[0][k][j][i] += 2.0 * dt * rhs[0][k][j][i];
                    u_stage[1][k][j][i] += 2.0 * dt * rhs[1][k][j][i];
                    u_stage[2][k][j][i] += 2.0 * dt * rhs[2][k][j][i];
                    u_stage[3][k][j][i] += 2.0 * dt * rhs[3][k][j][i];
                    u_stage[4][k][j][i] += 2.0 * dt * rhs[4][k][j][i];

                    u[0][k][j][i] = u_save[0][k][j][i] + 0.5 * dt * rhs[0][k][j][i];
                    u[1][k][j][i] = u_save[1][k][j][i] + 0.5 * dt * rhs[1][k][j][i];
                    u[2][k][j][i] = u_save[2][k][j][i] + 0.5 * dt * rhs[2][k][j][i];
                    u[3][k][j][i] = u_save[3][k][j][i] + 0.5 * dt * rhs[3][k][j][i];
                    u[4][k][j][i] = u_save[4][k][j][i] + 0.5 * dt * rhs[4][k][j][i];
                }
            }
        }
        break;
    case 2:  // k3: acc += 2*k3, u = U^n + k3
        for (PetscInt k = zs; k < zs + zm; k++) {
            for (PetscInt j = ys; j < ys + ym; j++) {
                for (PetscInt i = xs; i < xs + xm; i++) {
                    u_stage[0][k][j][i] += 2.0 * dt * rhs[0][k][j][i];
                    u_stage[1][k][j][i] += 2.0 * dt * rhs[1][k][j][i];
                    u_stage[2][k][j][i] += 2.0 * dt * rhs[2][k][j][i];
                    u_stage[3][k][j][i] += 2.0 * dt * rhs[3][k][j][i];
                    u_stage[4][k][j][i] += 2.0 * dt * rhs[4][k][j][i];

                    u[0][k][j][i] = u_save[0][k][j][i] + dt * rhs[0][k][j][i];
                    u[1][k][j][i] = u_save[1][k][j][i] + dt * rhs[1][k][j][i];
                    u[2][k][j][i] = u_save[2][k][j][i] + dt * rhs[2][k][j][i];
                    u[3][k][j][i] = u_save[3][k][j][i] + dt * rhs[3][k][j][i];
                    u[4][k][j][i] = u_save[4][k][j][i] + dt * rhs[4][k][j][i];
                }
            }
        }
        break;
    case 3:  // k4: acc += k4, u = U^n + acc/6
        {
            PetscReal inv6 = 1.0 / 6.0;
            for (PetscInt k = zs; k < zs + zm; k++) {
                for (PetscInt j = ys; j < ys + ym; j++) {
                    for (PetscInt i = xs; i < xs + xm; i++) {
                        u_stage[0][k][j][i] += dt * rhs[0][k][j][i];
                        u_stage[1][k][j][i] += dt * rhs[1][k][j][i];
                        u_stage[2][k][j][i] += dt * rhs[2][k][j][i];
                        u_stage[3][k][j][i] += dt * rhs[3][k][j][i];
                        u_stage[4][k][j][i] += dt * rhs[4][k][j][i];

                        u[0][k][j][i] = u_save[0][k][j][i] + u_stage[0][k][j][i] * inv6;
                        u[1][k][j][i] = u_save[1][k][j][i] + u_stage[1][k][j][i] * inv6;
                        u[2][k][j][i] = u_save[2][k][j][i] + u_stage[2][k][j][i] * inv6;
                        u[3][k][j][i] = u_save[3][k][j][i] + u_stage[3][k][j][i] * inv6;
                        u[4][k][j][i] = u_save[4][k][j][i] + u_stage[4][k][j][i] * inv6;
                    }
                }
            }
        }
        break;
    }
    return 0;
}

// ====================================================================
// RK5Integrator — 5 阶段 4 阶 low-storage Runge-Kutta
//
// 使用 Carpenter & Kennedy (1994) 的 5-stage 4th-order 低存储方案。
// 只需要 2 组寄存器（u 和 u_stage），不需要保留中间 stage 历史。
//
// 算法（2N 存储）：
//   Q  = U^n
//   dQ = 0
//   for stage i = 0..4:
//     dQ = A[i] * dQ + dt * R(Q)
//     Q  = Q + B[i] * dQ
//   U^(n+1) = Q
//
// 系数（Carpenter & Kennedy 1994, 5-stage 4th-order）：
//   A = [0, -0.4178904745, -1.1921516946, -1.6977846924, -1.5141834442]
//   B = [0.1496590211, 0.3792103130, 0.8229550294, 0.6994504559, 0.1530572479]
//
// dQ 用 u_stage 存储，Q 用 u 存储，u_save 存 U^n 供参考。
// ====================================================================
PetscErrorCode RK5Integrator::step(
    Mesh* mesh, RHSAssembler* rhs_asm,
    PetscReal dt, PetscReal gamma,
    PetscReal*** u[5], PetscReal*** rhs[5],
    PetscReal*** u_save[5], PetscReal*** u_stage[5],
    GhostFillFunc fillGhosts)
{
    PetscErrorCode ierr;
    DM da = mesh->getDM();

    PetscInt xs, ys, zs, xm, ym, zm;
    getLocalRange(da, xs, ys, zs, xm, ym, zm);

    // Carpenter & Kennedy (1994) 5-stage 4th-order low-storage RK 系数
    const PetscReal A[5] = {
        0.0,
        -0.4178904745,
        -1.1921516946,
        -1.6977846924,
        -1.5141834442
    };
    const PetscReal B[5] = {
        0.1496590211,
        0.3792103130,
        0.8229550294,
        0.6994504559,
        0.1530572479
    };

    // 初始化：Q = U^n（u 中已包含），dQ = 0
    for (PetscInt k = zs; k < zs + zm; k++) {
        for (PetscInt j = ys; j < ys + ym; j++) {
            for (PetscInt i = xs; i < xs + xm; i++) {
                // 顺便保存 U^n
                u_save[0][k][j][i] = u[0][k][j][i];
                u_save[1][k][j][i] = u[1][k][j][i];
                u_save[2][k][j][i] = u[2][k][j][i];
                u_save[3][k][j][i] = u[3][k][j][i];
                u_save[4][k][j][i] = u[4][k][j][i];
                // dQ = 0
                u_stage[0][k][j][i] = 0.0;
                u_stage[1][k][j][i] = 0.0;
                u_stage[2][k][j][i] = 0.0;
                u_stage[3][k][j][i] = 0.0;
                u_stage[4][k][j][i] = 0.0;
            }
        }
    }

    // Low-storage RK 循环：5 个 stage
    for (int stage = 0; stage < 5; ++stage) {
        ierr = fillGhosts(); CHKERRQ(ierr);
        ierr = rhs_asm->computeRHS(mesh, gamma, u, rhs); CHKERRQ(ierr);

        PetscReal a = A[stage];
        PetscReal b = B[stage];

        for (PetscInt k = zs; k < zs + zm; k++) {
            for (PetscInt j = ys; j < ys + ym; j++) {
                for (PetscInt i = xs; i < xs + xm; i++) {
                    for (int c = 0; c < 5; ++c) {
                        // dQ = a * dQ + dt * R(Q)
                        u_stage[c][k][j][i] = a * u_stage[c][k][j][i]
                                            + dt * rhs[c][k][j][i];
                        // Q = Q + b * dQ
                        u[c][k][j][i] += b * u_stage[c][k][j][i];
                    }
                }
            }
        }
    }
    // u 现在 = U^(n+1)

    return 0;
}

PetscErrorCode RK5Integrator::initStageLoop(
    Mesh* mesh, PetscReal*** u[5],
    PetscReal*** u_save[5], PetscReal*** u_stage[5])
{
    DM da = mesh->getDM();
    PetscInt xs, ys, zs, xm, ym, zm;
    getLocalRange(da, xs, ys, zs, xm, ym, zm);

    // 保存 U^n → u_save，清零 dQ 累加器 u_stage
    for (PetscInt k = zs; k < zs + zm; k++) {
        for (PetscInt j = ys; j < ys + ym; j++) {
            for (PetscInt i = xs; i < xs + xm; i++) {
                u_save[0][k][j][i] = u[0][k][j][i];
                u_save[1][k][j][i] = u[1][k][j][i];
                u_save[2][k][j][i] = u[2][k][j][i];
                u_save[3][k][j][i] = u[3][k][j][i];
                u_save[4][k][j][i] = u[4][k][j][i];

                u_stage[0][k][j][i] = 0.0;
                u_stage[1][k][j][i] = 0.0;
                u_stage[2][k][j][i] = 0.0;
                u_stage[3][k][j][i] = 0.0;
                u_stage[4][k][j][i] = 0.0;
            }
        }
    }
    return 0;
}

PetscErrorCode RK5Integrator::applyStage(
    Mesh* mesh, PetscReal dt,
    PetscReal*** u[5], PetscReal*** rhs[5],
    PetscReal*** u_save[5], PetscReal*** u_stage[5],
    int stage)
{
    DM da = mesh->getDM();
    PetscInt xs, ys, zs, xm, ym, zm;
    getLocalRange(da, xs, ys, zs, xm, ym, zm);

    (void)u_save;

    // Carpenter & Kennedy (1994) 5-stage 4th-order low-storage RK 系数
    const PetscReal A[5] = {
        0.0,
        -0.4178904745,
        -1.1921516946,
        -1.6977846924,
        -1.5141834442
    };
    const PetscReal B[5] = {
        0.1496590211,
        0.3792103130,
        0.8229550294,
        0.6994504559,
        0.1530572479
    };

    PetscReal a = A[stage];
    PetscReal b = B[stage];

    // dQ = a * dQ + dt * R(Q)
    // Q  = Q + b * dQ
    for (PetscInt k = zs; k < zs + zm; k++) {
        for (PetscInt j = ys; j < ys + ym; j++) {
            for (PetscInt i = xs; i < xs + xm; i++) {
                for (int c = 0; c < 5; ++c) {
                    u_stage[c][k][j][i] = a * u_stage[c][k][j][i]
                                        + dt * rhs[c][k][j][i];
                    u[c][k][j][i] += b * u_stage[c][k][j][i];
                }
            }
        }
    }
    return 0;
}

// ====================================================================
// createTimeIntegrator — 简单工厂
// ====================================================================
TimeIntegrator* createTimeIntegrator(const std::string& name)
{
    if      (name == "ForwardEuler") return new ForwardEulerIntegrator();
    else if (name == "RK3")          return new RK3Integrator();
    else if (name == "RK4")          return new RK4Integrator();
    else if (name == "RK5")          return new RK5Integrator();
    return nullptr;
}
