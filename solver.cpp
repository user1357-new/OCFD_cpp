#include "solver.h"
#include "mesh.h"
#include "mesh_mutiblock.h"
#include "BC_filler.h"
#include "BC_ghost_filler.h"
#include "sim_config.h"
#include "flux_scheme.h"
#include "flux_derivative.h"
#include "time_integrator.h"
#include <petsc.h>
#include <cmath>
#include <iostream>
#include <stdexcept>

// ====================================================================
// RHSAssembler
// ====================================================================
RHSAssembler::RHSAssembler(const LinearEulerFlux& flux, FluxDerivative* deriv)
    : flux_(flux), deriv_(deriv), work_ready_(false)
{
}

RHSAssembler::~RHSAssembler()
{
    cleanupWorkVectors();
}

PetscErrorCode RHSAssembler::setupWorkVectors(Mesh* mesh)
{
    PetscErrorCode ierr;
    DM da = mesh->getDM();

    cleanupWorkVectors();

    for (int c = 0; c < 5; ++c) {
        ierr = DMCreateLocalVector(da, &Fhat_local_[c]);  CHKERRQ(ierr);
        ierr = DMCreateLocalVector(da, &Ghat_local_[c]);  CHKERRQ(ierr);
        ierr = DMCreateLocalVector(da, &Hhat_local_[c]);  CHKERRQ(ierr);
        ierr = DMCreateLocalVector(da, &dFhat_local_[c]); CHKERRQ(ierr);
        ierr = DMCreateLocalVector(da, &dGhat_local_[c]); CHKERRQ(ierr);
        ierr = DMCreateLocalVector(da, &dHhat_local_[c]); CHKERRQ(ierr);
    }
    work_ready_ = true;
    return 0;
}

void RHSAssembler::cleanupWorkVectors()
{
    if (!work_ready_) return;
    for (int c = 0; c < 5; ++c) {
        if (Fhat_local_[c])  { VecDestroy(&Fhat_local_[c]);  Fhat_local_[c]  = nullptr; }
        if (Ghat_local_[c])  { VecDestroy(&Ghat_local_[c]);  Ghat_local_[c]  = nullptr; }
        if (Hhat_local_[c])  { VecDestroy(&Hhat_local_[c]);  Hhat_local_[c]  = nullptr; }
        if (dFhat_local_[c]) { VecDestroy(&dFhat_local_[c]); dFhat_local_[c] = nullptr; }
        if (dGhat_local_[c]) { VecDestroy(&dGhat_local_[c]); dGhat_local_[c] = nullptr; }
        if (dHhat_local_[c]) { VecDestroy(&dHhat_local_[c]); dHhat_local_[c] = nullptr; }
    }
    work_ready_ = false;
}

PetscErrorCode RHSAssembler::computeRHS(Mesh* mesh, PetscReal gamma,
                                         PetscReal*** u[5], PetscReal*** rhs[5])
{
    PetscErrorCode ierr;
    DM da = mesh->getDM();

    if (!work_ready_) {
        ierr = setupWorkVectors(mesh); CHKERRQ(ierr);
    }

    // ---- 获取度量系数 local 数组 ----
    PetscReal ***akx, ***aky, ***akz;
    PetscReal ***aix, ***aiy, ***aiz;
    PetscReal ***asx, ***asy, ***asz;
    PetscReal ***ajac;
    DMDALocalInfo info;
    ierr = mesh->getCellMetricArrays(akx,aky,akz, aix,aiy,aiz, asx,asy,asz, ajac, info);
    CHKERRQ(ierr);

    PetscInt xs = info.xs, ys = info.ys, zs = info.zs;
    PetscInt xm = info.xm, ym = info.ym, zm = info.zm;
    PetscInt gxs = info.gxs, gys = info.gys, gzs = info.gzs;
    PetscInt gxm = info.gxm, gym = info.gym, gzm = info.gzm;

    // ---- 获取通量临时数组（全部用 global 索引）----
    PetscReal ***fhat[5], ***ghat[5], ***hhat[5];
    PetscReal ***dfhat[5], ***dghat[5], ***dhhat[5];
    for (int c = 0; c < 5; ++c) {
        ierr = DMDAVecGetArray(da, Fhat_local_[c],  &fhat[c]);  CHKERRQ(ierr);
        ierr = DMDAVecGetArray(da, Ghat_local_[c],  &ghat[c]);  CHKERRQ(ierr);
        ierr = DMDAVecGetArray(da, Hhat_local_[c],  &hhat[c]);  CHKERRQ(ierr);
        ierr = DMDAVecGetArray(da, dFhat_local_[c], &dfhat[c]); CHKERRQ(ierr);
        ierr = DMDAVecGetArray(da, dGhat_local_[c], &dghat[c]); CHKERRQ(ierr);
        ierr = DMDAVecGetArray(da, dHhat_local_[c], &dhhat[c]); CHKERRQ(ierr);
    }

    // ---- 第 1 步：遍历所有格心（含 ghost），算物理通量 + 变换到计算空间 ----
    for (PetscInt k = gzs; k < gzs + gzm; k++) {
        for (PetscInt j = gys; j < gys + gym; j++) {
            for (PetscInt i = gxs; i < gxs + gxm; i++) {

                PetscReal U_cell[5] = {
                    u[0][k][j][i], u[1][k][j][i], u[2][k][j][i],
                    u[3][k][j][i], u[4][k][j][i]
                };

                PetscReal F[5], G[5], H[5];
                flux_.computeFlux(U_cell, F, G, H);

                PetscReal kx = akx[k][j][i], ky = aky[k][j][i], kz = akz[k][j][i];
                PetscReal ix = aix[k][j][i], iy = aiy[k][j][i], iz = aiz[k][j][i];
                PetscReal sx = asx[k][j][i], sy = asy[k][j][i], sz = asz[k][j][i];

                for (int c = 0; c < 5; ++c) {
                    fhat[c][k][j][i] = kx * F[c] + ky * G[c] + kz * H[c];
                    ghat[c][k][j][i] = ix * F[c] + iy * G[c] + iz * H[c];
                    hhat[c][k][j][i] = sx * F[c] + sy * G[c] + sz * H[c];
                }
            }
        }
    }

    // ---- 第 2 步：逐分量求三个方向导数 ----
    for (int c = 0; c < 5; ++c) {
        deriv_->deriv_xi(da,   fhat[c], dfhat[c]);
        deriv_->deriv_eta(da,  ghat[c], dghat[c]);
        deriv_->deriv_zeta(da, hhat[c], dhhat[c]);
    }

    // ---- 第 3 步：组装 RHS = -(dF̂/dξ + dĜ/dη + dĤ/dζ) / J ----
    for (PetscInt k = zs; k < zs + zm; k++) {
        for (PetscInt j = ys; j < ys + ym; j++) {
            for (PetscInt i = xs; i < xs + xm; i++) {

                PetscReal invJ = 1.0 / ajac[k][j][i];

                for (int c = 0; c < 5; ++c) {
                    rhs[c][k][j][i] = -(dfhat[c][k][j][i]
                                      + dghat[c][k][j][i]
                                      + dhhat[c][k][j][i]) * invJ;
                }
            }
        }
    }

    // ---- 恢复数组 ----
    for (int c = 0; c < 5; ++c) {
        ierr = DMDAVecRestoreArray(da, Fhat_local_[c],  &fhat[c]);  CHKERRQ(ierr);
        ierr = DMDAVecRestoreArray(da, Ghat_local_[c],  &ghat[c]);  CHKERRQ(ierr);
        ierr = DMDAVecRestoreArray(da, Hhat_local_[c],  &hhat[c]);  CHKERRQ(ierr);
        ierr = DMDAVecRestoreArray(da, dFhat_local_[c], &dfhat[c]); CHKERRQ(ierr);
        ierr = DMDAVecRestoreArray(da, dGhat_local_[c], &dghat[c]); CHKERRQ(ierr);
        ierr = DMDAVecRestoreArray(da, dHhat_local_[c], &dhhat[c]); CHKERRQ(ierr);
    }

    ierr = mesh->restoreCellMetricArrays(akx,aky,akz, aix,aiy,aiz, asx,asy,asz, ajac);
    CHKERRQ(ierr);

    return 0;
}

// ====================================================================
// computeBlockCFL — 线性化 Euler（静止基底）
//
// 扰动变量 U' = [ρ', ρ₀u', ρ₀v', ρ₀w', p']
// 总速度 = 扰动速度（u₀=0）, 总声速从基态 + 扰动计算。
// ====================================================================
PetscReal computeBlockCFL(Mesh* mesh, PetscReal*** u[5],
                           PetscReal cfl_target, PetscReal gamma,
                           PetscReal base_rho, PetscReal base_p)
{
    DMDALocalInfo info;
    DM da = mesh->getDM();
    DMDAGetLocalInfo(da, &info);

    PetscInt xs = info.xs, ys = info.ys, zs = info.zs;
    PetscInt xm = info.xm, ym = info.ym, zm = info.zm;

    PetscReal ***akx, ***aky, ***akz;
    PetscReal ***aix, ***aiy, ***aiz;
    PetscReal ***asx, ***asy, ***asz;
    PetscReal ***ajac;
    mesh->getCellMetricArrays(akx,aky,akz, aix,aiy,aiz, asx,asy,asz, ajac, info);

    PetscReal max_lambda = 0.0;
    PetscReal inv_base_rho = 1.0 / base_rho;

    for (PetscInt k = zs; k < zs + zm; k++) {
        for (PetscInt j = ys; j < ys + ym; j++) {
            for (PetscInt i = xs; i < xs + xm; i++) {

                // 扰动速度 u' = ρ₀u' / ρ₀
                PetscReal vel_u = u[1][k][j][i] * inv_base_rho;
                PetscReal vel_v = u[2][k][j][i] * inv_base_rho;
                PetscReal vel_w = u[3][k][j][i] * inv_base_rho;

                // 总密度和总压力
                PetscReal rho_tot = base_rho + u[0][k][j][i];   // ρ₀ + ρ'
                PetscReal p_tot   = base_p   + u[4][k][j][i];   // p₀ + p'
                if (rho_tot < 1e-10) rho_tot = 1e-10;
                if (p_tot   < 1e-10) p_tot   = 1e-10;
                PetscReal c = std::sqrt(gamma * p_tot / rho_tot);

                PetscReal U_xi  = akx[k][j][i]*vel_u + aky[k][j][i]*vel_v + akz[k][j][i]*vel_w;
                PetscReal U_eta = aix[k][j][i]*vel_u + aiy[k][j][i]*vel_v + aiz[k][j][i]*vel_w;
                PetscReal U_zeta= asx[k][j][i]*vel_u + asy[k][j][i]*vel_v + asz[k][j][i]*vel_w;

                PetscReal r_xi  = std::abs(U_xi)  + c * std::sqrt(akx[k][j][i]*akx[k][j][i] + aky[k][j][i]*aky[k][j][i] + akz[k][j][i]*akz[k][j][i]);
                PetscReal r_eta = std::abs(U_eta) + c * std::sqrt(aix[k][j][i]*aix[k][j][i] + aiy[k][j][i]*aiy[k][j][i] + aiz[k][j][i]*aiz[k][j][i]);
                PetscReal r_zeta= std::abs(U_zeta)+ c * std::sqrt(asx[k][j][i]*asx[k][j][i] + asy[k][j][i]*asy[k][j][i] + asz[k][j][i]*asz[k][j][i]);

                PetscReal lambda = (r_xi + r_eta + r_zeta) / ajac[k][j][i];

                if (lambda > max_lambda) max_lambda = lambda;
            }
        }
    }

    mesh->restoreCellMetricArrays(akx,aky,akz, aix,aiy,aiz, asx,asy,asz, ajac);

    PetscReal global_max;
    MPI_Allreduce(&max_lambda, &global_max, 1, MPIU_REAL, MPI_MAX,
                  mesh->getComm());

    if (global_max < 1.0e-12) return 1.0e-6;
    return cfl_target / global_max;
}

// ====================================================================
// computeBlockResidual
// ====================================================================
PetscReal computeBlockResidual(Mesh* mesh, PetscReal*** rhs[5])
{
    DMDALocalInfo info;
    DM da = mesh->getDM();
    DMDAGetLocalInfo(da, &info);

    PetscInt xs = info.xs, ys = info.ys, zs = info.zs;
    PetscInt xm = info.xm, ym = info.ym, zm = info.zm;

    PetscReal sum_sq = 0.0;
    PetscInt n_cells = 0;

    for (PetscInt k = zs; k < zs + zm; k++) {
        for (PetscInt j = ys; j < ys + ym; j++) {
            for (PetscInt i = xs; i < xs + xm; i++) {
                for (int c = 0; c < 5; ++c) {
                    PetscReal val = rhs[c][k][j][i];
                    sum_sq += val * val;
                }
                n_cells++;
            }
        }
    }

    PetscReal global_sum_sq;
    MPI_Allreduce(&sum_sq, &global_sum_sq, 1, MPIU_REAL, MPI_SUM,
                  mesh->getComm());

    PetscInt global_n;
    MPI_Allreduce(&n_cells, &global_n, 1, MPIU_INT, MPI_SUM,
                  mesh->getComm());

    return std::sqrt(global_sum_sq / (PetscReal)(global_n * 5));
}

// ====================================================================
// runEulerSolver — 求解器主循环（线性化 Euler 方程）
// ====================================================================
PetscErrorCode runEulerSolver(
    MultiBlockMesh& multiMesh,
    FluxDerivative* deriv,
    TimeIntegrator* time_int,
    PetscReal cfl, PetscReal gamma,
    PetscInt max_steps, PetscInt output_interval,
    GhostFillFunc fillAllGhosts,
    PetscReal c0sq, PetscReal base_rho, PetscReal base_p)
{
    PetscErrorCode ierr;
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    PetscInt nBlocks = multiMesh.getNumBlocks();

    LinearEulerFlux flux(c0sq);
    RHSAssembler rhsAsm(flux, deriv);

    // ---- 每块分配 RHS + stage 临时 local Vec ----
    struct BlockWork {
        Vec rhs_vec[5];
        Vec u_save_vec[5];
        Vec u_stage_vec[5];
    };
    std::vector<BlockWork> works(nBlocks);

    for (PetscInt b = 0; b < nBlocks; ++b) {
        Mesh* blk = multiMesh.getBlock(b);
        if (!blk) continue;
        DM da = blk->getDM();
        for (int c = 0; c < 5; ++c) {
            ierr = DMCreateLocalVector(da, &works[b].rhs_vec[c]);     CHKERRQ(ierr);
            ierr = DMCreateLocalVector(da, &works[b].u_save_vec[c]);  CHKERRQ(ierr);
            ierr = DMCreateLocalVector(da, &works[b].u_stage_vec[c]); CHKERRQ(ierr);
        }
        rhsAsm.setupWorkVectors(blk);
    }

    // ---- 主时间循环 ----
    PetscReal t = 0.0;
    for (PetscInt step = 0; step < max_steps; ++step) {

        // 1. 计算全局时间步长（CFL 只用内点，不需要 ghost）
        PetscReal dt = 1.0e10;
        for (PetscInt b = 0; b < nBlocks; ++b) {
            Mesh* blk = multiMesh.getBlock(b);
            if (!blk) continue;
            PetscReal*** u[5];
            DMDALocalInfo info;
            ierr = blk->getLocalUArrays(u, info); CHKERRQ(ierr);
            PetscReal dt_b = computeBlockCFL(blk, u, cfl, gamma, base_rho, base_p);
            ierr = blk->restoreLocalUArrays(u); CHKERRQ(ierr);
            if (dt_b < dt) dt = dt_b;
        }

        // 2. 初始化所有块的 stage 数据（保存 U^n，清零累加器）
        for (PetscInt b = 0; b < nBlocks; ++b) {
            Mesh* blk = multiMesh.getBlock(b);
            if (!blk) continue;
            DM da = blk->getDM();

            PetscReal*** u[5];
            DMDALocalInfo info;
            ierr = blk->getLocalUArrays(u, info); CHKERRQ(ierr);

            PetscReal*** u_save[5];
            PetscReal*** u_stage[5];
            for (int c = 0; c < 5; ++c) {
                ierr = DMDAVecGetArray(da, works[b].u_save_vec[c],  &u_save[c]);  CHKERRQ(ierr);
                ierr = DMDAVecGetArray(da, works[b].u_stage_vec[c], &u_stage[c]); CHKERRQ(ierr);
            }

            ierr = time_int->initStageLoop(blk, u, u_save, u_stage); CHKERRQ(ierr);

            for (int c = 0; c < 5; ++c) {
                ierr = DMDAVecRestoreArray(da, works[b].u_save_vec[c],  &u_save[c]);  CHKERRQ(ierr);
                ierr = DMDAVecRestoreArray(da, works[b].u_stage_vec[c], &u_stage[c]); CHKERRQ(ierr);
            }

            ierr = blk->restoreLocalUArrays(u); CHKERRQ(ierr);
        }

        // 3. Stage-outer, block-inner 时间推进
        //    所有块在同一 stage 级别一起推进，保证块间 ghost 数据同步
        int nStages = time_int->numStages();
        for (int stage = 0; stage < nStages; ++stage) {

            // 填充所有块的 ghost（所有块处于同一 stage 级别）
            ierr = fillAllGhosts(); CHKERRQ(ierr);

            // 对该 stage，逐块计算 RHS 并按 RK 公式更新
            for (PetscInt b = 0; b < nBlocks; ++b) {
                Mesh* blk = multiMesh.getBlock(b);
                if (!blk) continue;
                DM da = blk->getDM();

                PetscReal*** u[5];
                DMDALocalInfo info;
                ierr = blk->getLocalUArrays(u, info); CHKERRQ(ierr);

                PetscReal*** rhs[5];
                PetscReal*** u_save[5];
                PetscReal*** u_stage[5];
                for (int c = 0; c < 5; ++c) {
                    ierr = DMDAVecGetArray(da, works[b].rhs_vec[c],     &rhs[c]);     CHKERRQ(ierr);
                    ierr = DMDAVecGetArray(da, works[b].u_save_vec[c],  &u_save[c]);  CHKERRQ(ierr);
                    ierr = DMDAVecGetArray(da, works[b].u_stage_vec[c], &u_stage[c]); CHKERRQ(ierr);
                }

                // 计算 RHS
                ierr = rhsAsm.computeRHS(blk, gamma, u, rhs); CHKERRQ(ierr);

                // 按 RK 公式更新 u（单 stage）
                ierr = time_int->applyStage(blk, dt, u, rhs, u_save, u_stage, stage); CHKERRQ(ierr);

                for (int c = 0; c < 5; ++c) {
                    ierr = DMDAVecRestoreArray(da, works[b].rhs_vec[c],     &rhs[c]);     CHKERRQ(ierr);
                    ierr = DMDAVecRestoreArray(da, works[b].u_save_vec[c],  &u_save[c]);  CHKERRQ(ierr);
                    ierr = DMDAVecRestoreArray(da, works[b].u_stage_vec[c], &u_stage[c]); CHKERRQ(ierr);
                }

                ierr = blk->restoreLocalUArrays(u); CHKERRQ(ierr);
                if (ierr) return ierr;
            }
        }

        // 4. 同步: local → global → local（保证 PETSc ghost 区正确）
        for (PetscInt b = 0; b < nBlocks; ++b) {
            Mesh* blk = multiMesh.getBlock(b);
            if (!blk) continue;
            ierr = blk->syncULocalToGlobal();   CHKERRQ(ierr);
            ierr = blk->syncUGlobalToLocal();   CHKERRQ(ierr);
        }

        t += dt;

        // 5. 输出 + 残差监控
        if (step % output_interval == 0 || step == max_steps - 1) {
            // 填充 ghost 后再算残差，确保块间交界面 ghost 正确
            ierr = fillAllGhosts(); CHKERRQ(ierr);

            PetscReal total_res = 0.0;
            for (PetscInt b = 0; b < nBlocks; ++b) {
                Mesh* blk = multiMesh.getBlock(b);
                if (!blk) continue;

                PetscReal*** u_arr[5];
                DMDALocalInfo info;
                ierr = blk->getLocalUArrays(u_arr, info); CHKERRQ(ierr);

                PetscReal*** rhs_arr[5];
                for (int c = 0; c < 5; ++c) {
                    ierr = DMDAVecGetArray(blk->getDM(), works[b].rhs_vec[c], &rhs_arr[c]);
                    CHKERRQ(ierr);
                }

                rhsAsm.computeRHS(blk, gamma, u_arr, rhs_arr);
                PetscReal res_b = computeBlockResidual(blk, rhs_arr);

                for (int c = 0; c < 5; ++c) {
                    ierr = DMDAVecRestoreArray(blk->getDM(), works[b].rhs_vec[c], &rhs_arr[c]);
                    CHKERRQ(ierr);
                }

                ierr = blk->restoreLocalUArrays(u_arr); CHKERRQ(ierr);

                if (res_b > total_res) total_res = res_b;
            }

            if (rank == 0) {
                std::cout << "Step " << step << "  t = " << t
                          << "  dt = " << dt
                          << "  res = " << total_res << std::endl;
            }

            multiMesh.ExportAllFlowToTecplot("flow_step_" + std::to_string(step));
        }
    }

    // ---- 清理 ----
    for (PetscInt b = 0; b < nBlocks; ++b) {
        for (int c = 0; c < 5; ++c) {
            if (works[b].rhs_vec[c])     { VecDestroy(&works[b].rhs_vec[c]);     works[b].rhs_vec[c]     = nullptr; }
            if (works[b].u_save_vec[c])  { VecDestroy(&works[b].u_save_vec[c]);  works[b].u_save_vec[c]  = nullptr; }
            if (works[b].u_stage_vec[c]) { VecDestroy(&works[b].u_stage_vec[c]); works[b].u_stage_vec[c] = nullptr; }
        }
    }
    rhsAsm.cleanupWorkVectors();

    return 0;
}

// ====================================================================
// setupAndRunSolver — 一站式求解器入口
//
// 封装 BC 注册、初始 ghost 填充、组件创建、fillGhosts lambda 和
// runEulerSolver 调用，main 函数只需调用此一个接口。
// ====================================================================
PetscErrorCode setupAndRunSolver(
    MultiBlockMesh& multiMesh,
    const SimConfig& cfg,
    JacGhostExtentBC& ghostFiller)
{
    PetscErrorCode ierr;
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    PetscReal gamma    = cfg.getGamma();
    PetscReal base_rho = cfg.getBaseRho();
    PetscReal base_p   = cfg.getBaseP();
    PetscReal c0sq     = cfg.getBaseC0() * cfg.getBaseC0();

    // ================================================================
    // 1. 注册全部物理边界条件
    // ================================================================
    {
        PetscReal Ma    = cfg.getMach();
        PetscReal AoA   = cfg.getAttack()  * M_PI / 180.0;
        PetscReal beta  = cfg.getSideslip() * M_PI / 180.0;
        PetscReal rho_inf = 1.0;
        PetscReal u_inf   = std::cos(AoA) * std::cos(beta);
        PetscReal v_inf   = std::sin(AoA) * std::cos(beta);
        PetscReal w_inf   = std::sin(beta);
        PetscReal p_inf   = 1.0 / (gamma * Ma * Ma);

        BCFillerRegistry& reg = BCFillerRegistry::instance();
        reg.add("BCWall",
                std::unique_ptr<BCFaceFiller>(new BCWallFiller(gamma)));
        reg.add("BCAcousticWall",
                std::unique_ptr<BCFaceFiller>(new BCAcousticWallFiller(gamma, base_rho)));
        reg.add("BCInlet",
                std::unique_ptr<BCFaceFiller>(
                    new BCInletFiller(gamma,
                        cfg.getInletRho(), cfg.getInletU(),
                        cfg.getInletV(), cfg.getInletW(),
                        cfg.getInletP())));
        reg.add("BCFarfield",
                std::unique_ptr<BCFaceFiller>(
                    new BCFarfieldFiller(gamma, rho_inf, u_inf, v_inf, w_inf, p_inf)));
        reg.add("BCExtrapolate",
                std::unique_ptr<BCFaceFiller>(new BCExtrapolateFiller()));
    }

    // ================================================================
    // 2. 初始 U ghost 填充：块间抄数 → 物理 BC
    // ================================================================
    multiMesh.gatherFullFlow();
    multiMesh.exchangeDonorSlabsU();
    multiMesh.exchangeDonorSlabsUPeriodic();

    for (int b = 0; b < multiMesh.getNumBlocks(); ++b) {
        Mesh* blk = multiMesh.getBlock(b);
        if (!blk) continue;
        for (int face = 0; face < 6; ++face) {
            auto conn = multiMesh.findConnectedFace(b, face);
            if (conn.first >= 0) {
                ghostFiller.fillGhostOnFaceFromVecs(blk, face,
                    blk->getLocalUVecs(), &multiMesh, b, 3);
            }
        }
    }

    {
        BCFillerRegistry& reg = BCFillerRegistry::instance();
        for (int b = 0; b < multiMesh.getNumBlocks(); ++b) {
            Mesh* blk = multiMesh.getBlock(b);
            if (!blk) continue;
            for (int face = 0; face < 6; ++face) {
                if (multiMesh.findConnectedFace(b, face).first >= 0)
                    continue;
                std::string bc_type;
                for (const auto& fbc : multiMesh.getFaceBCs()) {
                    if (fbc.block == b && fbc.face == face)
                    { bc_type = fbc.bc_type; break; }
                }
                if (bc_type.empty()) continue;
                BCFaceFiller* filler = reg.get(bc_type);
                if (!filler) continue;
                PetscReal*** u[5];
                DMDALocalInfo info;
                blk->getLocalUArrays(u, info);
                filler->fillGhostFace(blk, face, u, info);
                blk->restoreLocalUArrays(u);
            }
        }
    }
    if (rank == 0) PetscPrintf(PETSC_COMM_WORLD, "Initial U ghost fill done.\n");

    // ================================================================
    // 3. 求解器（仅当 max_steps > 0 时运行）
    // ================================================================
    if (cfg.getMaxSteps() <= 0) return 0;

    FluxDerivative* deriv = createDerivative(cfg.getSpatialDerivative());
    if (!deriv)
        throw std::runtime_error("Unknown spatial_derivative: " + cfg.getSpatialDerivative());

    TimeIntegrator* time_int = createTimeIntegrator(cfg.getTimeIntegrator());
    if (!time_int)
        throw std::runtime_error("Unknown time_integrator: " + cfg.getTimeIntegrator());

    // ---- Ghost 填充回调（每个 Runge-Kutta 子步调用）----
    auto fillGhosts = [&]() -> PetscErrorCode {
        PetscErrorCode ierr2;
        // 1. local→global
        for (int b = 0; b < multiMesh.getNumBlocks(); ++b) {
            Mesh* blk = multiMesh.getBlock(b);
            if (!blk) continue;
            ierr2 = blk->syncULocalToGlobal(); CHKERRQ(ierr2);
        }
        // 2. 汇集流场 + 周期面 exchange
        multiMesh.gatherFullFlow();
        multiMesh.exchangeDonorSlabsUPeriodic();
        // 3. 块间连接面
        for (int b = 0; b < multiMesh.getNumBlocks(); ++b) {
            Mesh* blk = multiMesh.getBlock(b);
            if (!blk) continue;
            for (int face = 0; face < 6; ++face) {
                if (multiMesh.findConnectedFace(b, face).first >= 0) {
                    ghostFiller.fillGhostOnFaceFromVecs(blk, face,
                        blk->getLocalUVecs(), &multiMesh, b, 3);
                }
            }
        }
        // 4. 物理 BC
        {
            BCFillerRegistry& reg = BCFillerRegistry::instance();
            for (int b = 0; b < multiMesh.getNumBlocks(); ++b) {
                Mesh* blk = multiMesh.getBlock(b);
                if (!blk) continue;
                for (int face = 0; face < 6; ++face) {
                    if (multiMesh.findConnectedFace(b, face).first >= 0) continue;
                    std::string bc_type;
                    for (const auto& fbc : multiMesh.getFaceBCs()) {
                        if (fbc.block == b && fbc.face == face)
                        { bc_type = fbc.bc_type; break; }
                    }
                    if (bc_type.empty()) continue;
                    BCFaceFiller* filler = reg.get(bc_type);
                    if (!filler) continue;
                    PetscReal*** u[5];
                    DMDALocalInfo info;
                    blk->getLocalUArrays(u, info);
                    filler->fillGhostFace(blk, face, u, info);
                    blk->restoreLocalUArrays(u);
                }
            }
        }
        // 5. 边/角 ghost
        for (int b = 0; b < multiMesh.getNumBlocks(); ++b) {
            Mesh* blk = multiMesh.getBlock(b);
            if (!blk) continue;
            DMDALocalInfo info;
            PetscReal*** u[5];
            blk->getLocalUArrays(u, info);
            PetscInt nxg = blk->getNxGlobal();
            PetscInt nyg = blk->getNyGlobal();
            PetscInt nzg = blk->getNzGlobal();
            for (PetscInt k = info.gzs; k < info.gzs + info.gzm; ++k)
                for (PetscInt j = info.gys; j < info.gys + info.gym; ++j)
                    for (PetscInt i = info.gxs; i < info.gxs + info.gxm; ++i) {
                        int ndir = (i<0||i>=nxg)+(j<0||j>=nyg)+(k<0||k>=nzg);
                        if (ndir < 2) continue;
                        PetscInt ip = (i<0)?0:((i>=nxg)?nxg-1:i);
                        PetscInt jp = (j<0)?0:((j>=nyg)?nyg-1:j);
                        PetscInt kp = (k<0)?0:((k>=nzg)?nzg-1:k);
                        for (int c = 0; c < 5; ++c) u[c][k][j][i] = u[c][kp][jp][ip];
                    }
            blk->restoreLocalUArrays(u);
        }
        return 0;
    };

    ierr = runEulerSolver(multiMesh, deriv, time_int,
                          cfg.getCFL(), gamma,
                          cfg.getMaxSteps(), cfg.getOutputInterval(),
                          fillGhosts, c0sq, base_rho, base_p);
    delete time_int;
    delete deriv;
    if (ierr) return ierr;
    if (rank == 0) PetscPrintf(PETSC_COMM_WORLD, "Solver done.\n");

    return 0;
}
