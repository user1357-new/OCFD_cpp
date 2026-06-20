#include "mesh.h"
#include <petscdmda.h>
#include <petscvec.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <mpi.h>
#include <vector>
#include <string>

// ====================================================================
// 静态：获取 x 方向 k 阶中心差分系数（内联）
//   返回 coeff[0..k-1]：df/dx = Σ coeff[m] * (f[i+m+1] - f[i-m-1]) / (2h)
//   这里的 coeff 已含 1/(2h)
// ====================================================================
static inline void getCD2_coeff(PetscReal h, PetscReal *c) {
    c[0] = 1.0 / (2.0 * h);    // (f1 - f_{-1})
}
static inline void getCD4_coeff(PetscReal h, PetscReal *c) {
    c[0] =  8.0 / (12.0 * h);  //  (8/12)/h  → (f1-f_{-1})
    c[1] = -1.0 / (12.0 * h);  // (-1/12)/h  → (f2-f_{-2})
}
static inline void getCD6_coeff(PetscReal h, PetscReal *c) {
    c[0] =  45.0 / (60.0 * h);
    c[1] =  -9.0 / (60.0 * h);
    c[2] =   1.0 / (60.0 * h);
}
static inline void getCD8_coeff(PetscReal h, PetscReal *c) {
    c[0] = 0.8 / h;
    c[1] = -0.2 / h;
    c[2] = 3.80952380952380952e-2 / h;
    c[3] = -3.571428571428571428e-3 / h;
}

// 2 阶单向差分（边界 i=0 用前向，i=max 用后向）
static inline PetscReal oneside2(PetscReal fm2, PetscReal fm1,
                                  PetscReal f0, PetscReal fp1, PetscReal fp2,
                                  int side, PetscReal h)
{
    // side=0: 左边界，前向  df = (-3f0 + 4f1 - f2) / (2h)
    // side=1: 右边界，后向  df = ( 3f0 - 4fm1 + fm2) / (2h)
    if (side == 0)
        return (-3.0 * f0 + 4.0 * fp1 - fp2) / (2.0 * h);
    else
        return ( 3.0 * f0 - 4.0 * fm1 + fm2) / (2.0 * h);
}

// ====================================================================
// compute_derivative_x — 格心坐标 x 方向导数
// f  : 来自格心 local Vec（含 ghost），全局索引
// df : 用 0-based 局部索引 [kl][jl][il]
// ====================================================================
void Mesh::compute_derivative_x(PetscReal ***f, PetscReal ***df)
{
    DMDALocalInfo info;
    DMDAGetLocalInfo(da, &info);

    // ★ 格心间距：N_c 个格心在 [0,1] 中，间距 = 1/N_c
    PetscReal hx = 1.0 / (PetscReal)(nx_global);

    PetscInt xs = info.xs, ys = info.ys, zs = info.zs;
    PetscInt xm = info.xm, ym = info.ym, zm = info.zm;

    // CD6/CD8 系数（均匀模式用）
    PetscReal cd6_c[3], cd8_c[4];
    if (Scheme_Vis == OCFD_Scheme_CD6) getCD6_coeff(hx, cd6_c);
    else                               getCD8_coeff(hx, cd8_c);

    for (PetscInt k = zs; k < zs + zm; k++) {
        PetscInt kl = k - zs;
        for (PetscInt j = ys; j < ys + ym; j++) {
            PetscInt jl = j - ys;
            for (PetscInt i = xs; i < xs + xm; i++) {
                PetscInt il = i - xs;

                if (metric_diff_type_ == METRIC_DIFF_UNIFORM) {
                    // ====== 模式 A：全场统一 CD6/CD8 ======
                    if (Scheme_Vis == OCFD_Scheme_CD6) {
                        df[kl][jl][il] = cd6_c[0]*(f[k][j][i+1]-f[k][j][i-1])
                                       + cd6_c[1]*(f[k][j][i+2]-f[k][j][i-2])
                                       + cd6_c[2]*(f[k][j][i+3]-f[k][j][i-3]);
                    } else {
                        df[kl][jl][il] = cd8_c[0]*(f[k][j][i+1]-f[k][j][i-1])
                                       + cd8_c[1]*(f[k][j][i+2]-f[k][j][i-2])
                                       + cd8_c[2]*(f[k][j][i+3]-f[k][j][i-3])
                                       + cd8_c[3]*(f[k][j][i+4]-f[k][j][i-4]);
                    }

                } else {
                    // ====== 模式 B：边界逐层降阶 ======
                    PetscInt dist = i;
                    if (nx_global - 1 - i < dist)
                        dist = nx_global - 1 - i;

                    switch (dist) {
                    case 0: {
                        // 第 0 层：2 阶单向
                        int side = (i == 0) ? 0 : 1;
                        if (side == 0)
                            df[kl][jl][il] = oneside2(
                                f[k][j][i], f[k][j][i],      // f_{-2}, f_{-1} 不存在，填 0
                                f[k][j][i], f[k][j][i+1], f[k][j][i+2], 0, hx);
                        else
                            df[kl][jl][il] = oneside2(
                                f[k][j][i-2], f[k][j][i-1],
                                f[k][j][i], f[k][j][i], f[k][j][i], 1, hx);
                        break;
                    }
                    case 1: {
                        // 第 1 层：2 阶中心
                        PetscReal c;
                        getCD2_coeff(hx, &c);
                        df[kl][jl][il] = c * (f[k][j][i+1] - f[k][j][i-1]);
                        break;
                    }
                    case 2: {
                        // 第 2 层：4 阶中心
                        PetscReal c[2];
                        getCD4_coeff(hx, c);
                        df[kl][jl][il] = c[0]*(f[k][j][i+1]-f[k][j][i-1])
                                       + c[1]*(f[k][j][i+2]-f[k][j][i-2]);
                        break;
                    }
                    case 3: {
                        // 第 3 层：CD6
                        PetscReal c[3];
                        getCD6_coeff(hx, c);
                        df[kl][jl][il] = c[0]*(f[k][j][i+1]-f[k][j][i-1])
                                       + c[1]*(f[k][j][i+2]-f[k][j][i-2])
                                       + c[2]*(f[k][j][i+3]-f[k][j][i-3]);
                        break;
                    }
                    default: {
                        // 第 4+ 层：用户选 CD6/CD8
                        if (Scheme_Vis == OCFD_Scheme_CD6) {
                            df[kl][jl][il] = cd6_c[0]*(f[k][j][i+1]-f[k][j][i-1])
                                           + cd6_c[1]*(f[k][j][i+2]-f[k][j][i-2])
                                           + cd6_c[2]*(f[k][j][i+3]-f[k][j][i-3]);
                        } else {
                            df[kl][jl][il] = cd8_c[0]*(f[k][j][i+1]-f[k][j][i-1])
                                           + cd8_c[1]*(f[k][j][i+2]-f[k][j][i-2])
                                           + cd8_c[2]*(f[k][j][i+3]-f[k][j][i-3])
                                           + cd8_c[3]*(f[k][j][i+4]-f[k][j][i-4]);
                        }
                        break;
                    }
                    } // switch
                } // reduced
            }
        }
    }
}

// ====================================================================
// compute_derivative_y
// ====================================================================
void Mesh::compute_derivative_y(PetscReal ***f, PetscReal ***df)
{
    DMDALocalInfo info;
    DMDAGetLocalInfo(da, &info);

    // ★ 格心间距：N_c 个格心在 [0,1] 中，间距 = 1/N_c
    PetscReal hy = 1.0 / (PetscReal)(ny_global);

    PetscInt xs = info.xs, ys = info.ys, zs = info.zs;
    PetscInt xm = info.xm, ym = info.ym, zm = info.zm;

    PetscReal cd6_c[3], cd8_c[4];
    if (Scheme_Vis == OCFD_Scheme_CD6) getCD6_coeff(hy, cd6_c);
    else                               getCD8_coeff(hy, cd8_c);

    for (PetscInt k = zs; k < zs + zm; k++) {
        PetscInt kl = k - zs;
        for (PetscInt j = ys; j < ys + ym; j++) {
            PetscInt jl = j - ys;
            for (PetscInt i = xs; i < xs + xm; i++) {
                PetscInt il = i - xs;

                if (metric_diff_type_ == METRIC_DIFF_UNIFORM) {
                    if (Scheme_Vis == OCFD_Scheme_CD6) {
                        df[kl][jl][il] = cd6_c[0]*(f[k][j+1][i]-f[k][j-1][i])
                                       + cd6_c[1]*(f[k][j+2][i]-f[k][j-2][i])
                                       + cd6_c[2]*(f[k][j+3][i]-f[k][j-3][i]);
                    } else {
                        df[kl][jl][il] = cd8_c[0]*(f[k][j+1][i]-f[k][j-1][i])
                                       + cd8_c[1]*(f[k][j+2][i]-f[k][j-2][i])
                                       + cd8_c[2]*(f[k][j+3][i]-f[k][j-3][i])
                                       + cd8_c[3]*(f[k][j+4][i]-f[k][j-4][i]);
                    }
                } else {
                    PetscInt dist = j;
                    if (ny_global - 1 - j < dist)
                        dist = ny_global - 1 - j;

                    switch (dist) {
                    case 0: {
                        int side = (j == 0) ? 0 : 1;
                        if (side == 0)
                            df[kl][jl][il] = oneside2(0,0, f[k][j][i], f[k][j+1][i], f[k][j+2][i], 0, hy);
                        else
                            df[kl][jl][il] = oneside2(f[k][j-2][i], f[k][j-1][i], f[k][j][i], 0, 0, 1, hy);
                        break;
                    }
                    case 1: {
                        PetscReal c; getCD2_coeff(hy, &c);
                        df[kl][jl][il] = c * (f[k][j+1][i] - f[k][j-1][i]);
                        break;
                    }
                    case 2: {
                        PetscReal c[2]; getCD4_coeff(hy, c);
                        df[kl][jl][il] = c[0]*(f[k][j+1][i]-f[k][j-1][i])
                                       + c[1]*(f[k][j+2][i]-f[k][j-2][i]);
                        break;
                    }
                    case 3: {
                        PetscReal c[3]; getCD6_coeff(hy, c);
                        df[kl][jl][il] = c[0]*(f[k][j+1][i]-f[k][j-1][i])
                                       + c[1]*(f[k][j+2][i]-f[k][j-2][i])
                                       + c[2]*(f[k][j+3][i]-f[k][j-3][i]);
                        break;
                    }
                    default:
                        if (Scheme_Vis == OCFD_Scheme_CD6)
                            df[kl][jl][il] = cd6_c[0]*(f[k][j+1][i]-f[k][j-1][i])
                                           + cd6_c[1]*(f[k][j+2][i]-f[k][j-2][i])
                                           + cd6_c[2]*(f[k][j+3][i]-f[k][j-3][i]);
                        else
                            df[kl][jl][il] = cd8_c[0]*(f[k][j+1][i]-f[k][j-1][i])
                                           + cd8_c[1]*(f[k][j+2][i]-f[k][j-2][i])
                                           + cd8_c[2]*(f[k][j+3][i]-f[k][j-3][i])
                                           + cd8_c[3]*(f[k][j+4][i]-f[k][j-4][i]);
                    }
                }
            }
        }
    }
}

// ====================================================================
// compute_derivative_z
// ====================================================================
void Mesh::compute_derivative_z(PetscReal ***f, PetscReal ***df)
{
    DMDALocalInfo info;
    DMDAGetLocalInfo(da, &info);

    // ★ 格心间距：N_c 个格心在 [0,1] 中，间距 = 1/N_c
    PetscReal hz = 1.0 / (PetscReal)(nz_global);

    PetscInt xs = info.xs, ys = info.ys, zs = info.zs;
    PetscInt xm = info.xm, ym = info.ym, zm = info.zm;

    PetscReal cd6_c[3], cd8_c[4];
    if (Scheme_Vis == OCFD_Scheme_CD6) getCD6_coeff(hz, cd6_c);
    else                               getCD8_coeff(hz, cd8_c);

    for (PetscInt k = zs; k < zs + zm; k++) {
        PetscInt kl = k - zs;
        for (PetscInt j = ys; j < ys + ym; j++) {
            PetscInt jl = j - ys;
            for (PetscInt i = xs; i < xs + xm; i++) {
                PetscInt il = i - xs;

                if (metric_diff_type_ == METRIC_DIFF_UNIFORM) {
                    if (Scheme_Vis == OCFD_Scheme_CD6) {
                        df[kl][jl][il] = cd6_c[0]*(f[k+1][j][i]-f[k-1][j][i])
                                       + cd6_c[1]*(f[k+2][j][i]-f[k-2][j][i])
                                       + cd6_c[2]*(f[k+3][j][i]-f[k-3][j][i]);
                    } else {
                        df[kl][jl][il] = cd8_c[0]*(f[k+1][j][i]-f[k-1][j][i])
                                       + cd8_c[1]*(f[k+2][j][i]-f[k-2][j][i])
                                       + cd8_c[2]*(f[k+3][j][i]-f[k-3][j][i])
                                       + cd8_c[3]*(f[k+4][j][i]-f[k-4][j][i]);
                    }
                } else {
                    PetscInt dist = k;
                    if (nz_global - 1 - k < dist)
                        dist = nz_global - 1 - k;

                    switch (dist) {
                    case 0: {
                        int side = (k == 0) ? 0 : 1;
                        if (side == 0)
                            df[kl][jl][il] = oneside2(0,0, f[k][j][i], f[k+1][j][i], f[k+2][j][i], 0, hz);
                        else
                            df[kl][jl][il] = oneside2(f[k-2][j][i], f[k-1][j][i], f[k][j][i], 0,0, 1, hz);
                        break;
                    }
                    case 1: {
                        PetscReal c; getCD2_coeff(hz, &c);
                        df[kl][jl][il] = c * (f[k+1][j][i] - f[k-1][j][i]);
                        break;
                    }
                    case 2: {
                        PetscReal c[2]; getCD4_coeff(hz, c);
                        df[kl][jl][il] = c[0]*(f[k+1][j][i]-f[k-1][j][i])
                                       + c[1]*(f[k+2][j][i]-f[k-2][j][i]);
                        break;
                    }
                    case 3: {
                        PetscReal c[3]; getCD6_coeff(hz, c);
                        df[kl][jl][il] = c[0]*(f[k+1][j][i]-f[k-1][j][i])
                                       + c[1]*(f[k+2][j][i]-f[k-2][j][i])
                                       + c[2]*(f[k+3][j][i]-f[k-3][j][i]);
                        break;
                    }
                    default:
                        if (Scheme_Vis == OCFD_Scheme_CD6)
                            df[kl][jl][il] = cd6_c[0]*(f[k+1][j][i]-f[k-1][j][i])
                                           + cd6_c[1]*(f[k+2][j][i]-f[k-2][j][i])
                                           + cd6_c[2]*(f[k+3][j][i]-f[k-3][j][i]);
                        else
                            df[kl][jl][il] = cd8_c[0]*(f[k+1][j][i]-f[k-1][j][i])
                                           + cd8_c[1]*(f[k+2][j][i]-f[k-2][j][i])
                                           + cd8_c[2]*(f[k+3][j][i]-f[k-3][j][i])
                                           + cd8_c[3]*(f[k+4][j][i]-f[k-4][j][i]);
                    }
                }
            }
        }
    }
}

// ====================================================================
// computeCellJacobianFromFilledGhosts — 用已填 ghost 的格心坐标算格心度量系数
// ====================================================================
PetscErrorCode Mesh::computeCellJacobianFromFilledGhosts()
{
    PetscErrorCode ierr;

    if (!cell_pool_initialized) {
        ierr = ensureCellLocalVectors(); CHKERRQ(ierr);
    }

    DMDALocalInfo info;
    PetscReal ***axx, ***ayy, ***azz;
    ierr = getCellCoordinateArrays(axx, ayy, azz, info); CHKERRQ(ierr);

    PetscReal ***akx, ***aky, ***akz;
    PetscReal ***aix, ***aiy, ***aiz;
    PetscReal ***asx, ***asy, ***asz;
    PetscReal ***ajac;
    ierr = DMDAVecGetArray(da, Akx, &akx); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Aky, &aky); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Akz, &akz); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Aix, &aix); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Aiy, &aiy); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Aiz, &aiz); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Asx, &asx); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Asy, &asy); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Asz, &asz); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Ajac, &ajac); CHKERRQ(ierr);

    PetscInt xs = info.xs, ys = info.ys, zs = info.zs;
    PetscInt xm = info.xm, ym = info.ym, zm = info.zm;

    // 分配临时导数数组
    PetscReal ***xi, ***xj, ***xk;
    PetscReal ***yi, ***yj, ***yk;
    PetscReal ***zi, ***zj, ***zk;

    PetscMalloc1(zm, &xi); PetscMalloc1(zm, &xj); PetscMalloc1(zm, &xk);
    PetscMalloc1(zm, &yi); PetscMalloc1(zm, &yj); PetscMalloc1(zm, &yk);
    PetscMalloc1(zm, &zi); PetscMalloc1(zm, &zj); PetscMalloc1(zm, &zk);

    for (PetscInt kl = 0; kl < zm; kl++) {
        PetscMalloc1(ym, &xi[kl]); PetscMalloc1(ym, &xj[kl]); PetscMalloc1(ym, &xk[kl]);
        PetscMalloc1(ym, &yi[kl]); PetscMalloc1(ym, &yj[kl]); PetscMalloc1(ym, &yk[kl]);
        PetscMalloc1(ym, &zi[kl]); PetscMalloc1(ym, &zj[kl]); PetscMalloc1(ym, &zk[kl]);

        for (PetscInt jl = 0; jl < ym; jl++) {
            PetscMalloc1(xm, &xi[kl][jl]); PetscMalloc1(xm, &xj[kl][jl]); PetscMalloc1(xm, &xk[kl][jl]);
            PetscMalloc1(xm, &yi[kl][jl]); PetscMalloc1(xm, &yj[kl][jl]); PetscMalloc1(xm, &yk[kl][jl]);
            PetscMalloc1(xm, &zi[kl][jl]); PetscMalloc1(xm, &zj[kl][jl]); PetscMalloc1(xm, &zk[kl][jl]);
        }
    }

    compute_derivative_x(axx, xi);
    compute_derivative_x(ayy, yi);
    compute_derivative_x(azz, zi);
    compute_derivative_y(axx, xj);
    compute_derivative_y(ayy, yj);
    compute_derivative_y(azz, zj);
    compute_derivative_z(axx, xk);
    compute_derivative_z(ayy, yk);
    compute_derivative_z(azz, zk);

    // 组装雅可比
    for (PetscInt k = zs; k < zs + zm; k++) {
        PetscInt kl = k - zs;
        for (PetscInt j = ys; j < ys + ym; j++) {
            PetscInt jl = j - ys;
            for (PetscInt i = xs; i < xs + xm; i++) {
                PetscInt il = i - xs;

                PetscReal xi1 = xi[kl][jl][il], xj1 = xj[kl][jl][il], xk1 = xk[kl][jl][il];
                PetscReal yi1 = yi[kl][jl][il], yj1 = yj[kl][jl][il], yk1 = yk[kl][jl][il];
                PetscReal zi1 = zi[kl][jl][il], zj1 = zj[kl][jl][il], zk1 = zk[kl][jl][il];

                PetscReal Jac1 = 1.0 / (xi1*yj1*zk1 + yi1*zj1*xk1 + zi1*xj1*yk1
                                      - zi1*yj1*xk1 - yi1*xj1*zk1 - xi1*zj1*yk1);

                ajac[k][j][i] = Jac1;
                akx[k][j][i]  = Jac1 * (yj1*zk1 - zj1*yk1);
                aky[k][j][i]  = Jac1 * (zj1*xk1 - xj1*zk1);
                akz[k][j][i]  = Jac1 * (xj1*yk1 - yj1*xk1);
                aix[k][j][i]  = Jac1 * (yk1*zi1 - zk1*yi1);
                aiy[k][j][i]  = Jac1 * (zk1*xi1 - xk1*zi1);
                aiz[k][j][i]  = Jac1 * (xk1*yi1 - yk1*xi1);
                asx[k][j][i]  = Jac1 * (yi1*zj1 - zi1*yj1);
                asy[k][j][i]  = Jac1 * (zi1*xj1 - xi1*zj1);
                asz[k][j][i]  = Jac1 * (xi1*yj1 - yi1*xj1);

                if (Jac1 < 0) {
                    PetscPrintf(comm, " Jacobian < 0 !!! , Jac=%f\n", Jac1);
                }
            }
        }
    }

    // 释放临时数组
    for (PetscInt kl = 0; kl < zm; kl++) {
        for (PetscInt jl = 0; jl < ym; jl++) {
            PetscFree(xi[kl][jl]); PetscFree(xj[kl][jl]); PetscFree(xk[kl][jl]);
            PetscFree(yi[kl][jl]); PetscFree(yj[kl][jl]); PetscFree(yk[kl][jl]);
            PetscFree(zi[kl][jl]); PetscFree(zj[kl][jl]); PetscFree(zk[kl][jl]);
        }
        PetscFree(xi[kl]); PetscFree(xj[kl]); PetscFree(xk[kl]);
        PetscFree(yi[kl]); PetscFree(yj[kl]); PetscFree(yk[kl]);
        PetscFree(zi[kl]); PetscFree(zj[kl]); PetscFree(zk[kl]);
    }
    PetscFree(xi); PetscFree(xj); PetscFree(xk);
    PetscFree(yi); PetscFree(yj); PetscFree(yk);
    PetscFree(zi); PetscFree(zj); PetscFree(zk);

    // 恢复
    ierr = DMDAVecRestoreArray(da, Akx, &akx); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Aky, &aky); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Akz, &akz); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Aix, &aix); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Aiy, &aiy); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Aiz, &aiz); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Asx, &asx); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Asy, &asy); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Asz, &asz); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Ajac, &ajac); CHKERRQ(ierr);

    ierr = restoreCellCoordinateArrays(axx, ayy, azz); CHKERRQ(ierr);

    // 同步格心 global 度量 → local
    ierr = syncCellGlobalToLocal(); CHKERRQ(ierr);

    return 0;
}
