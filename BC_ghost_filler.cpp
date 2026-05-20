#include "BC_ghost_filler.h"
#include "mesh.h"
#include "mesh_mutiblock.h"
#include <petscdmda.h>
#include <fstream>
#include <sstream>
#include <mpi.h>
#include <cmath>
#include <iostream>

// ====================================================================
// fillGhostCellOnFace - 基类版本（模板方法）
// 获取持久化坐标 local vector，分别对 x,y,z 调用 assignGhostOnFace
// ====================================================================
PetscErrorCode GhostCellFiller::fillGhostCellOnFace(Mesh* mesh, int face,
                                                     MultiBlockMesh* all_blocks)
{
    // 统一入口：取坐标 Vec → 填充 → 返回
    return fillGhostOnFaceFromVecs(mesh, face,
                                   mesh->getLocalCoordinateVecs(), all_blocks);
}

// ====================================================================
// fillArraysOnFace - 对任意一组数组指针做 ghost 外推
// 不负责获取/释放数组，调用者自己管理生命周期
// ====================================================================
PetscErrorCode GhostCellFiller::fillArraysOnFace(
    Mesh* mesh, int face,
    const std::vector<PetscReal***>& arrays,
    const DMDALocalInfo& info,
    MultiBlockMesh* all_blocks)
{
    PetscErrorCode ierr;
    if (!mesh) return 0;
    if (face < 0 || face > 5) return 1;

    for (size_t n = 0; n < arrays.size(); ++n) {
        if (!arrays[n]) continue;  // 跳过空指针
        ierr = assignGhostOnFace(mesh, face, arrays[n], info, all_blocks);
        CHKERRQ(ierr);
    }
    return 0;
}

// ====================================================================
// fillGhostOnFaceFromVecs - 从 local Vec 列表自动获取/释放
// 典型用法：填充物理量 ghost（rho, u, v, w, p, T ...）
// ====================================================================
PetscErrorCode GhostCellFiller::fillGhostOnFaceFromVecs(
    Mesh* mesh, int face,
    const std::vector<Vec>& localVecs,
    MultiBlockMesh* all_blocks)
{
    PetscErrorCode ierr;
    if (!mesh) return 0;
    if (face < 0 || face > 5) return 1;
    if (localVecs.empty()) return 0;

    DM da = mesh->getDM();
    DMDALocalInfo info;
    ierr = DMDAGetLocalInfo(da, &info); CHKERRQ(ierr);

    // 获取所有数组指针
    std::vector<PetscReal***> arrays(localVecs.size(), nullptr);
    for (size_t n = 0; n < localVecs.size(); ++n) {
        if (localVecs[n] == nullptr) continue;
        ierr = DMDAVecGetArray(da, localVecs[n], &arrays[n]); CHKERRQ(ierr);
    }

    // 批量填充
    ierr = fillArraysOnFace(mesh, face, arrays, info, all_blocks); CHKERRQ(ierr);

    // 释放所有数组指针
    for (size_t n = 0; n < localVecs.size(); ++n) {
        if (localVecs[n] == nullptr || arrays[n] == nullptr) continue;
        ierr = DMDAVecRestoreArray(da, localVecs[n], &arrays[n]); CHKERRQ(ierr);
    }
    return 0;
}

// ====================================================================
// JacGhostExtentBC 构造函数
// ====================================================================
JacGhostExtentBC::JacGhostExtentBC(PetscInt gtype, PetscInt lap)
    : GhostCellFiller(lap)
{
    for (int f = 0; f < 6; ++f)
        ghost_cell_face_[f] = gtype;
}

// ====================================================================
// JacGhostExtentBC 版本（填充 ghost坐标/度量系数）
// gtype=1: 一阶线性外推
// gtype=2: 二阶镜像外推
// ====================================================================
PetscErrorCode JacGhostExtentBC::fillGhostCellOnFace(Mesh* mesh, int face,
                                                      MultiBlockMesh* all_blocks)
{
    PetscErrorCode ierr;
    if (!mesh) return 0;
    if (face < 0 || face > 5) return 1;

    int gtype = ghost_cell_face_[face];
    if (gtype == 0) return 0;

    // 坐标：走基类（内部调 fillGhostOnFaceFromVecs）
    ierr = GhostCellFiller::fillGhostCellOnFace(mesh, face, all_blocks);
    CHKERRQ(ierr);

    // 度量系数：走同一条路径（Vec → fillGhostOnFaceFromVecs → assignGhostOnFace）
    ierr = fillGhostOnFaceFromVecs(mesh, face,
                                   mesh->getLocalMetricVecs(), all_blocks);
    CHKERRQ(ierr);
    return 0;
}

// ====================================================================
// 具体的 ghost 外推实现算法
// gtype=1: 一阶线性外推
// gtype=2: 二阶镜像外推
// ====================================================================
PetscErrorCode JacGhostExtentBC::assignGhostOnFace(Mesh* mesh,
                                                    int face,
                                                    PetscReal*** arr,
                                                    const DMDALocalInfo& info,
                                                    MultiBlockMesh* all_blocks)
{
    int gtype = ghost_cell_face_[face];
    if (gtype == 0) return 0;

    PetscInt nxg = mesh->getNxGlobal();
    PetscInt nyg = mesh->getNyGlobal();
    PetscInt nzg = mesh->getNzGlobal();

    // ---- 获取该面的 ghost 区域索引范围（已裁剪到本 rank）----
    int i_min, i_max, j_min, j_max, k_min, k_max;
    GetGhostRange(face, lap_, nxg, nyg, nzg, info,
                  i_min, i_max, j_min, j_max, k_min, k_max);

    // 本 rank 不拥有该面的任何 ghost 点
    if (i_min > i_max || j_min > j_max || k_min > k_max)
        return 0;

    // ============== gtype=1: 一阶线性外推 ==============
    if (gtype == 1) {
        for (PetscInt k = k_min; k <= k_max; ++k) {
            for (PetscInt j = j_min; j <= j_max; ++j) {
                for (PetscInt i = i_min; i <= i_max; ++i) {
                    // 跳过物理域点（GetGhostRange 只给 ghost 范围，但安全检查）
                    if (i >= 0 && i < nxg && j >= 0 && j < nyg && k >= 0 && k < nzg)
                        continue;

                    // 确定边界点和距离
                    PetscInt ib, jb, kb;
                    PetscReal dx = 0, dy = 0, dz = 0;
                    switch (face) {
                    case 0: ib = 0;       jb = j; kb = k; dx = (PetscReal)(-i);                break;
                    case 1: ib = nxg-1;   jb = j; kb = k; dx = (PetscReal)(i - (nxg - 1));     break;
                    case 2: ib = i; jb = 0;       kb = k; dy = (PetscReal)(-j);                break;
                    case 3: ib = i; jb = nyg-1;   kb = k; dy = (PetscReal)(j - (nyg - 1));     break;
                    case 4: ib = i; jb = j; kb = 0;       dz = (PetscReal)(-k);                break;
                    case 5: ib = i; jb = j; kb = nzg-1;   dz = (PetscReal)(k - (nzg - 1));     break;
                    default: continue;
                    }

                    // 确定内点（向物理域内部走一层）
                    PetscInt i2 = ib, j2 = jb, k2 = kb;
                    switch (face) {
                    case 0: i2 = ib + 1; break;
                    case 1: i2 = ib - 1; break;
                    case 2: j2 = jb + 1; break;
                    case 3: j2 = jb - 1; break;
                    case 4: k2 = kb + 1; break;
                    case 5: k2 = kb - 1; break;
                    }

                    PetscReal dVal = arr[kb][jb][ib] - arr[k2][j2][i2];
                    PetscReal dist = dx + dy + dz;
                    arr[k][j][i] = arr[kb][jb][ib] + dist * dVal;
                }
            }
        }
        return 0;
    }

    // ============== gtype=2: 二阶镜像外推 ==============
    if (gtype == 2) {
        for (PetscInt k = k_min; k <= k_max; ++k) {
            for (PetscInt j = j_min; j <= j_max; ++j) {
                for (PetscInt i = i_min; i <= i_max; ++i) {
                    if (i >= 0 && i < nxg && j >= 0 && j < nyg && k >= 0 && k < nzg)
                        continue;

                    // 计算镜像点索引
                    PetscInt i_mirror, j_mirror, k_mirror;
                    switch (face) {
                    case 0: i_mirror = -i;                     j_mirror = j; k_mirror = k; break;
                    case 1: i_mirror = 2*(nxg-1) - i;          j_mirror = j; k_mirror = k; break;
                    case 2: i_mirror = i; j_mirror = -j;                     k_mirror = k; break;
                    case 3: i_mirror = i; j_mirror = 2*(nyg-1) - j;         k_mirror = k; break;
                    case 4: i_mirror = i; j_mirror = j; k_mirror = -k;                    break;
                    case 5: i_mirror = i; j_mirror = j; k_mirror = 2*(nzg-1) - k;        break;
                    default: continue;
                    }

                    // 边界点索引
                    PetscInt ib, jb, kb;
                    switch (face) {
                    case 0: ib = 0;       jb = j; kb = k; break;
                    case 1: ib = nxg - 1; jb = j; kb = k; break;
                    case 2: ib = i; jb = 0;       kb = k; break;
                    case 3: ib = i; jb = nyg - 1; kb = k; break;
                    case 4: ib = i; jb = j; kb = 0;       break;
                    case 5: ib = i; jb = j; kb = nzg - 1; break;
                    default: continue;
                    }

                    arr[k][j][i] = 2.0 * arr[kb][jb][ib]
                                 - arr[k_mirror][j_mirror][i_mirror];
                }
            }
        }
        return 0;
    }

    return 0;
}

// ====================================================================
// 选取 ghost 区域索引范围
// ====================================================================
void GhostCellFiller::GetGhostRange(int face, int LAP,
                                    int NX, int NY, int NZ,
                                    const DMDALocalInfo& info,
                                    int& i_min, int& i_max,
                                    int& j_min, int& j_max,
                                    int& k_min, int& k_max)
{
    i_min = -LAP;   i_max = NX + LAP - 1;
    j_min = -LAP;   j_max = NY + LAP - 1;
    k_min = -LAP;   k_max = NZ + LAP - 1;

    switch (face) {
    case FACE_LEFT:
        i_min = -LAP;   i_max = -1;
        j_min = 0;      j_max = NY - 1;
        k_min = 0;      k_max = NZ - 1;
        break;
    case FACE_RIGHT:
        i_min = NX;     i_max = NX + LAP - 1;
        j_min = 0;      j_max = NY - 1;
        k_min = 0;      k_max = NZ - 1;
        break;
    case FACE_BOTTOM:
        i_min = 0;      i_max = NX - 1;
        j_min = -LAP;   j_max = -1;
        k_min = 0;      k_max = NZ - 1;
        break;
    case FACE_TOP:
        i_min = 0;      i_max = NX - 1;
        j_min = NY;     j_max = NY + LAP - 1;
        k_min = 0;      k_max = NZ - 1;
        break;
    case FACE_BACK:
        i_min = 0;      i_max = NX - 1;
        j_min = 0;      j_max = NY - 1;
        k_min = -LAP;   k_max = -1;
        break;
    case FACE_FRONT:
        i_min = 0;      i_max = NX - 1;
        j_min = 0;      j_max = NY - 1;
        k_min = NZ;     k_max = NZ + LAP - 1;
        break;
    }

    int i_lo = info.gxs,        i_hi = info.gxs + info.gxm - 1;
    int j_lo = info.gys,        j_hi = info.gys + info.gym - 1;
    int k_lo = info.gzs,        k_hi = info.gzs + info.gzm - 1;

    if (i_min < i_lo) i_min = i_lo;
    if (i_max > i_hi) i_max = i_hi;
    if (j_min < j_lo) j_min = j_lo;
    if (j_max > j_hi) j_max = j_hi;
    if (k_min < k_lo) k_min = k_lo;
    if (k_max > k_hi) k_max = k_hi;
}

// ====================================================================
// 输出包含虚网格的数据到 Tecplot 文件
// ====================================================================
PetscErrorCode GhostCellFiller::ExportGhostToTecplot(Mesh* mesh,
                                                     const std::string& base_filename)
{
    // ... 原样保留 ...
    // (内容太长不重复，见原始文件)
    PetscErrorCode ierr;
    if (!mesh) return 0;
    DM da = mesh->getDM();
    if (!da) return 0;

    MPI_Comm comm = mesh->getComm();
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    const int NVAR = 5;

    PetscInt LAP = mesh->getLAP();
    PetscInt nxg = mesh->getNxGlobal();
    PetscInt nyg = mesh->getNyGlobal();
    PetscInt nzg = mesh->getNzGlobal();
    PetscInt GI = nxg + 2 * LAP;
    PetscInt GJ = nyg + 2 * LAP;
    PetscInt GK = nzg + 2 * LAP;

    PetscReal ***axx, ***ayy, ***azz;
    DMDALocalInfo info;
    ierr = mesh->getLocalCoordinateArrays(axx, ayy, azz, info); CHKERRQ(ierr);

    PetscReal ***akx, ***ajac;
    PetscReal ***d1, ***d2, ***d3, ***d4, ***d5, ***d6, ***d7, ***d8;
    DMDALocalInfo info_m;
    ierr = mesh->getLocalMetricArrays(akx, d1, d2, d3, d4, d5, d6, d7, d8, ajac, info_m); CHKERRQ(ierr);

    PetscInt send_sz[3] = {info.gxm, info.gym, info.gzm};
    PetscInt send_off[3] = {info.gxs, info.gys, info.gzs};

    PetscInt(*all_sz)[3] = nullptr;
    PetscInt(*all_off)[3] = nullptr;
    if (rank == 0) {
        all_sz  = new PetscInt[nprocs][3];
        all_off = new PetscInt[nprocs][3];
    }
    MPI_Gather(send_sz, 3, MPI_INT, all_sz, 3, MPI_INT, 0, comm);
    MPI_Gather(send_off, 3, MPI_INT, all_off, 3, MPI_INT, 0, comm);

    PetscInt nlocal = info.gxm * info.gym * info.gzm;
    PetscReal *buf = new PetscReal[nlocal * NVAR];
    PetscInt p = 0;
    for (PetscInt k = info.gzs; k < info.gzs + info.gzm; ++k)
        for (PetscInt j = info.gys; j < info.gys + info.gym; ++j)
            for (PetscInt i = info.gxs; i < info.gxs + info.gxm; ++i) {
                buf[p++] = axx[k][j][i];
                buf[p++] = ayy[k][j][i];
                buf[p++] = azz[k][j][i];
                buf[p++] = akx[k][j][i];
                buf[p++] = akx[k][j][i] / ajac[k][j][i];   // akx1 = akx / ajac
            }

    int *recvcounts = nullptr, *displs = nullptr;
    PetscReal *all_buf = nullptr;
    if (rank == 0) {
        recvcounts = new int[nprocs];
        displs     = new int[nprocs];
        int total = 0;
        for (int i = 0; i < nprocs; ++i) {
            recvcounts[i] = all_sz[i][0] * all_sz[i][1] * all_sz[i][2] * NVAR;
            displs[i] = total;
            total += recvcounts[i];
        }
        all_buf = new PetscReal[total];
    }
    MPI_Gatherv(buf, nlocal * NVAR, MPI_DOUBLE, all_buf, recvcounts, displs, MPI_DOUBLE, 0, comm);

    // rank 0 写入文件
    if (rank == 0) {
        PetscReal *grid = new PetscReal[GI * GJ * GK * NVAR]();
        for (int r = 0; r < nprocs; ++r) {
            PetscInt *sz  = all_sz[r];
            PetscInt *off = all_off[r];
            PetscReal *src = all_buf + displs[r];
            PetscInt idx = 0;
            for (PetscInt kl = 0; kl < sz[2]; ++kl) {
                PetscInt k = off[2] + kl;
                for (PetscInt jl = 0; jl < sz[1]; ++jl) {
                    PetscInt j = off[1] + jl;
                    for (PetscInt il = 0; il < sz[0]; ++il) {
                        PetscInt i = off[0] + il;
                        PetscInt dst = ((k + LAP) * GJ * GI + (j + LAP) * GI + (i + LAP)) * NVAR;
                        for (int v = 0; v < NVAR; ++v)
                            grid[dst + v] = src[idx++];
                    }
                }
            }
        }

        std::string fname = base_filename + ".dat";
        std::ofstream out(fname);
        out << "TITLE=\"" << base_filename << "\"\n"
            << "VARIABLES=\"X\",\"Y\",\"Z\",\"Akx\",\"Akx1\"\n"
            << "ZONE T=\"" << base_filename << "\",I=" << GI
            << ",J=" << GJ << ",K=" << GK << ",F=POINT\n";
        out.precision(8);
        out << std::scientific;
        PetscInt ntot = GI * GJ * GK;
        for (PetscInt n = 0; n < ntot; ++n)
            out << grid[n * NVAR]     << " "
                << grid[n * NVAR + 1] << " "
                << grid[n * NVAR + 2] << " "
                << grid[n * NVAR + 3] << " "
                << grid[n * NVAR + 4] << "\n";
        out.close();
        std::cout << "Exported: " << fname << " (" << ntot << " pts)\n";

        delete[] grid;
        delete[] recvcounts;
        delete[] displs;
        delete[] all_buf;
    } else {
        delete[] recvcounts;
        delete[] displs;
        delete[] all_buf;
    }

    delete[] buf;
    ierr = mesh->restoreLocalMetricArrays(akx, d1, d2, d3, d4, d5, d6, d7, d8, ajac); CHKERRQ(ierr);
    ierr = mesh->restoreLocalCoordinateArrays(axx, ayy, azz); CHKERRQ(ierr);

    // 清理 all_sz/all_off
    delete[] all_sz;
    delete[] all_off;

    return 0;
}

// ====================================================================
// CompositeBC：每个面委托给不同的 BC 对象（暂时没有用）
// ====================================================================
PetscErrorCode CompositeBC::assignGhostOnFace(Mesh* mesh,
                                              int face,
                                              PetscReal*** arr,
                                              const DMDALocalInfo& info,
                                              MultiBlockMesh* all_blocks)
{
    if (face < 0 || face > 5) return 1;
    if (!face_bc_[face]) return 0;  // 该面未指定 BC，跳过
    return face_bc_[face]->assignGhostOnFace(mesh, face, arr, info, all_blocks);
}
