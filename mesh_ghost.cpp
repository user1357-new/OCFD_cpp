#include "mesh.h"
#include <petscdmda.h>
#include <petscvec.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <mpi.h>
#include <vector>
#include <string>
// MPI边界交换接口（留空给您实现）
void Mesh::exchange_boundary_xyz(Vec &f)
{
    // TODO: 您需要实现MPI边界交换
    // 这里可以使用PETSc的VecGhostUpdate或自定义MPI通信
}
// 注册边界条件处理函数
void Mesh::RegisterBoundaryConditionHandler(std::function<void()> handler)
{
    boundary_condition_handler = handler;
}
// ====================================================================
// fillEdgeCornerArray - 内部辅助：对单个数组填充边角虚网格
//
// 依赖链：面 ghost（1个ghost方向）→ 边 ghost（2个方向）→ 角 ghost（3个方向）
//
// gtype=1 (常数延拓)：边和角直接 clamp 到物理域边界
// gtype=2 (镜像外推)：先填边（面 ghost 为输入），再填角（边 ghost 为输入）
//                     边/角均为各 ghost 方向镜向外推后取平均
// ====================================================================
static void fillEdgeCornerArray(
    PetscReal*** arr,
    const DMDALocalInfo& info,
    PetscInt nxg, PetscInt nyg, PetscInt nzg,
    int gtype)
{
    PetscInt gxs = info.gxs, gxe = info.gxs + info.gxm;
    PetscInt gys = info.gys, gye = info.gys + info.gym;
    PetscInt gzs = info.gzs, gze = info.gzs + info.gzm;

    auto nGhostDirs = [&](PetscInt i, PetscInt j, PetscInt k) -> int {
        int n = 0;
        if (i < 0 || i >= nxg) n++;
        if (j < 0 || j >= nyg) n++;
        if (k < 0 || k >= nzg) n++;
        return n;
    };

    auto isPhysical = [&](PetscInt i, PetscInt j, PetscInt k) -> bool {
        return i >= 0 && i < nxg && j >= 0 && j < nyg && k >= 0 && k < nzg;
    };

    auto clamp = [&](PetscInt v, PetscInt nmax) -> PetscInt {
        if (v < 0) return 0;
        if (v >= nmax) return nmax - 1;
        return v;
    };

    // ================ gtype=1: 一阶线性外推（与面一致）================
    if (gtype == 1) {

        // ---------- Pass 1: 边（2个ghost方向）----------
        for (PetscInt k = gzs; k < gze; ++k) {
            for (PetscInt j = gys; j < gye; ++j) {
                for (PetscInt i = gxs; i < gxe; ++i) {
                    if (isPhysical(i, j, k)) continue;
                    if (nGhostDirs(i, j, k) != 2) continue;

                    PetscReal sum = 0.0;
                    int cnt = 0;

                    if (i < 0 || i >= nxg) {
                        PetscInt ib = (i < 0) ? 0 : nxg - 1;
                        PetscInt i_inner = (i < 0) ? 1 : nxg - 2;
                        PetscReal dist = (i < 0) ? (PetscReal)(-i) : (PetscReal)(i - (nxg - 1));
                        sum += arr[k][j][ib] + dist * (arr[k][j][ib] - arr[k][j][i_inner]);
                        cnt++;
                    }
                    if (j < 0 || j >= nyg) {
                        PetscInt jb = (j < 0) ? 0 : nyg - 1;
                        PetscInt j_inner = (j < 0) ? 1 : nyg - 2;
                        PetscReal dist = (j < 0) ? (PetscReal)(-j) : (PetscReal)(j - (nyg - 1));
                        sum += arr[k][jb][i] + dist * (arr[k][jb][i] - arr[k][j_inner][i]);
                        cnt++;
                    }
                    if (k < 0 || k >= nzg) {
                        PetscInt kb = (k < 0) ? 0 : nzg - 1;
                        PetscInt k_inner = (k < 0) ? 1 : nzg - 2;
                        PetscReal dist = (k < 0) ? (PetscReal)(-k) : (PetscReal)(k - (nzg - 1));
                        sum += arr[kb][j][i] + dist * (arr[kb][j][i] - arr[k_inner][j][i]);
                        cnt++;
                    }

                    arr[k][j][i] = sum / (PetscReal)cnt;
                }
            }
        }

        // ---------- Pass 2: 角（3个ghost方向）----------
        for (PetscInt k = gzs; k < gze; ++k) {
            for (PetscInt j = gys; j < gye; ++j) {
                for (PetscInt i = gxs; i < gxe; ++i) {
                    if (isPhysical(i, j, k)) continue;
                    if (nGhostDirs(i, j, k) != 3) continue;

                    PetscReal sum = 0.0;

                    // 镜向 i
                    {
                        PetscInt ib = (i < 0) ? 0 : nxg - 1;
                        PetscInt i_inner = (i < 0) ? 1 : nxg - 2;
                        PetscReal dist = (i < 0) ? (PetscReal)(-i) : (PetscReal)(i - (nxg - 1));
                        sum += arr[k][j][ib] + dist * (arr[k][j][ib] - arr[k][j][i_inner]);
                    }
                    // 镜向 j
                    {
                        PetscInt jb = (j < 0) ? 0 : nyg - 1;
                        PetscInt j_inner = (j < 0) ? 1 : nyg - 2;
                        PetscReal dist = (j < 0) ? (PetscReal)(-j) : (PetscReal)(j - (nyg - 1));
                        sum += arr[k][jb][i] + dist * (arr[k][jb][i] - arr[k][j_inner][i]);
                    }
                    // 镜向 k
                    {
                        PetscInt kb = (k < 0) ? 0 : nzg - 1;
                        PetscInt k_inner = (k < 0) ? 1 : nzg - 2;
                        PetscReal dist = (k < 0) ? (PetscReal)(-k) : (PetscReal)(k - (nzg - 1));
                        sum += arr[kb][j][i] + dist * (arr[kb][j][i] - arr[k_inner][j][i]);
                    }

                    arr[k][j][i] = sum / 3.0;
                }
            }
        }
        return;
    }
    // ================ gtype=2: 镜像外推 ================

    // ---------- Pass 1: 边（2个ghost方向）----------
    // 镜向点只有 1 个 ghost 方向（面 ghost），已填入有效值
    for (PetscInt k = gzs; k < gze; ++k) {
        for (PetscInt j = gys; j < gye; ++j) {
            for (PetscInt i = gxs; i < gxe; ++i) {
                if (isPhysical(i, j, k)) continue;
                if (nGhostDirs(i, j, k) != 2) continue;

                PetscReal sum = 0.0;
                int cnt = 0;

                if (i < 0 || i >= nxg) {
                    PetscInt ib = (i < 0) ? 0 : nxg - 1;
                    PetscInt im = (i < 0) ? -i : 2 * (nxg - 1) - i;
                    sum += 2.0 * arr[k][j][ib] - arr[k][j][im];
                    cnt++;
                }
                if (j < 0 || j >= nyg) {
                    PetscInt jb = (j < 0) ? 0 : nyg - 1;
                    PetscInt jm = (j < 0) ? -j : 2 * (nyg - 1) - j;
                    sum += 2.0 * arr[k][jb][i] - arr[k][jm][i];
                    cnt++;
                }
                if (k < 0 || k >= nzg) {
                    PetscInt kb = (k < 0) ? 0 : nzg - 1;
                    PetscInt km = (k < 0) ? -k : 2 * (nzg - 1) - k;
                    sum += 2.0 * arr[kb][j][i] - arr[km][j][i];
                    cnt++;
                }

                arr[k][j][i] = sum / (PetscReal)cnt;
            }
        }
    }

    // ---------- Pass 2: 角（3个ghost方向）----------
    // 镜向点有 2 个 ghost 方向（边 ghost），Pass 1 已填入有效值
    for (PetscInt k = gzs; k < gze; ++k) {
        for (PetscInt j = gys; j < gye; ++j) {
            for (PetscInt i = gxs; i < gxe; ++i) {
                if (isPhysical(i, j, k)) continue;
                if (nGhostDirs(i, j, k) != 3) continue;

                PetscReal sum = 0.0;

                // 镜向 i
                {
                    PetscInt ib = (i < 0) ? 0 : nxg - 1;
                    PetscInt im = (i < 0) ? -i : 2 * (nxg - 1) - i;
                    sum += 2.0 * arr[k][j][ib] - arr[k][j][im];
                }
                // 镜向 j
                {
                    PetscInt jb = (j < 0) ? 0 : nyg - 1;
                    PetscInt jm = (j < 0) ? -j : 2 * (nyg - 1) - j;
                    sum += 2.0 * arr[k][jb][i] - arr[k][jm][i];
                }
                // 镜向 k
                {
                    PetscInt kb = (k < 0) ? 0 : nzg - 1;
                    PetscInt km = (k < 0) ? -k : 2 * (nzg - 1) - k;
                    sum += 2.0 * arr[kb][j][i] - arr[km][j][i];
                }

                arr[k][j][i] = sum / 3.0;
            }
        }
    }
}

// ====================================================================
// fillAllEdgeAndCornerGhost - 对所有持久化数组填充边角虚网格
// ====================================================================
PetscErrorCode Mesh::fillAllEdgeAndCornerGhost(int coord_gtype, int metric_gtype)
{
    PetscErrorCode ierr;

    DMDALocalInfo info;
    ierr = DMDAGetLocalInfo(da, &info); CHKERRQ(ierr);

    PetscInt nxg = nx_global, nyg = ny_global, nzg = nz_global;

    // ---- 坐标：用 coord_gtype ----
    {
        PetscReal ***axx, ***ayy, ***azz;
        ierr = getLocalCoordinateArrays(axx, ayy, azz, info); CHKERRQ(ierr);
        fillEdgeCornerArray(axx, info, nxg, nyg, nzg, coord_gtype);
        fillEdgeCornerArray(ayy, info, nxg, nyg, nzg, coord_gtype);
        fillEdgeCornerArray(azz, info, nxg, nyg, nzg, coord_gtype);
        ierr = restoreLocalCoordinateArrays(axx, ayy, azz); CHKERRQ(ierr);
    }

    // ---- 度量系数：用 metric_gtype ----
    {
        PetscReal ***akx, ***aky, ***akz;
        PetscReal ***aix, ***aiy, ***aiz;
        PetscReal ***asx, ***asy, ***asz;
        PetscReal ***ajac;
        ierr = getLocalMetricArrays(akx, aky, akz, aix, aiy, aiz,
                                    asx, asy, asz, ajac, info); CHKERRQ(ierr);
        fillEdgeCornerArray(akx, info, nxg, nyg, nzg, metric_gtype);
        fillEdgeCornerArray(aky, info, nxg, nyg, nzg, metric_gtype);
        fillEdgeCornerArray(akz, info, nxg, nyg, nzg, metric_gtype);
        fillEdgeCornerArray(aix, info, nxg, nyg, nzg, metric_gtype);
        fillEdgeCornerArray(aiy, info, nxg, nyg, nzg, metric_gtype);
        fillEdgeCornerArray(aiz, info, nxg, nyg, nzg, metric_gtype);
        fillEdgeCornerArray(asx, info, nxg, nyg, nzg, metric_gtype);
        fillEdgeCornerArray(asy, info, nxg, nyg, nzg, metric_gtype);
        fillEdgeCornerArray(asz, info, nxg, nyg, nzg, metric_gtype);
        fillEdgeCornerArray(ajac, info, nxg, nyg, nzg, metric_gtype);
        ierr = restoreLocalMetricArrays(akx, aky, akz, aix, aiy, aiz,
                                        asx, asy, asz, ajac); CHKERRQ(ierr);
    }

    return 0;
}
