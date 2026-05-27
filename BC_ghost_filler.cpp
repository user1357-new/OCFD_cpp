#include "BC_ghost_filler.h"
#include "mesh.h"
#include "mesh_mutiblock.h"
#include <petscdmda.h>
#include <fstream>
#include <sstream>
#include <mpi.h>
#include <cmath>
#include <iostream>
#include <cstdio>

// ====================================================================
// fillGhostCellOnFace - 基类版本（模板方法）
// 获取持久化坐标 local vector，分别对 x,y,z 调用 assignGhostOnFace
// ====================================================================
PetscErrorCode GhostCellFiller::fillGhostCellOnFace(Mesh* mesh, int face,
                                                     MultiBlockMesh* all_blocks)
{
    // 统一入口：合并坐标 + 度量系数 local Vec → 统一走 fillGhostOnFaceFromVecs → assignGhostOnFace
    std::vector<Vec> all_vecs = mesh->getLocalCoordinateVecs();
    std::vector<Vec> metric_vecs = mesh->getLocalMetricVecs();
    all_vecs.insert(all_vecs.end(), metric_vecs.begin(), metric_vecs.end());
    return fillGhostOnFaceFromVecs(mesh, face, all_vecs, all_blocks);
}

// ====================================================================
// fillGhostOnFaceFromVecs — 唯一的 Vec 遍历器
//
// block_id < 0  → 外推：调 assignGhostOnFace
// block_id >= 0 → 抄数：调 assignGhostOnConnection
// ====================================================================
PetscErrorCode GhostCellFiller::fillGhostOnFaceFromVecs(
    Mesh* mesh, int face,
    const std::vector<Vec>& localVecs,
    MultiBlockMesh* all_blocks,
    int block_id,
    int comp_offset)
{
    PetscErrorCode ierr;
    if (!mesh) return 0;
    if (face < 0 || face > 5) return 1;
    if (localVecs.empty()) return 0;

    DM da = mesh->getDM();
    DMDALocalInfo info;
    ierr = DMDAGetLocalInfo(da, &info); CHKERRQ(ierr);

    for (size_t n = 0; n < localVecs.size(); ++n) {
        if (localVecs[n] == nullptr) continue;
        PetscReal*** arr = nullptr;
        ierr = DMDAVecGetArray(da, localVecs[n], &arr); CHKERRQ(ierr);

        if (block_id < 0) {
            // 外推
            ierr = assignGhostOnFace(mesh, face, arr, info, all_blocks);
        } else {
            // 抄数
            ierr = assignGhostOnConnection(mesh, block_id, face, arr, info,
                                           all_blocks,
                                           comp_offset + static_cast<int>(n));
        }
        CHKERRQ(ierr);
        ierr = DMDAVecRestoreArray(da, localVecs[n], &arr); CHKERRQ(ierr);
    }
    return 0;
}

// ====================================================================
// GhostCellFiller::assignGhostOnConnection — 基类默认 no-op
// ====================================================================
PetscErrorCode GhostCellFiller::assignGhostOnConnection(
    Mesh* mesh, int block_id, int face,
    PetscReal*** arr, const DMDALocalInfo& info,
    MultiBlockMesh* all_blocks, int comp_idx)
{
    (void)mesh; (void)block_id; (void)face;
    (void)arr; (void)info; (void)all_blocks; (void)comp_idx;
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

    int i_min, i_max, j_min, j_max, k_min, k_max;
    GetGhostRange(face, lap_, nxg, nyg, nzg, info,
                  i_min, i_max, j_min, j_max, k_min, k_max);

    if (i_min > i_max || j_min > j_max || k_min > k_max)
        return 0;

    // ============== gtype=1: 一阶线性外推 ==============
    if (gtype == 1) {
        for (PetscInt k = k_min; k <= k_max; ++k) {
            for (PetscInt j = j_min; j <= j_max; ++j) {
                for (PetscInt i = i_min; i <= i_max; ++i) {
                    if (i >= 0 && i < nxg && j >= 0 && j < nyg && k >= 0 && k < nzg)
                        continue;

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
// fillGhostCellOnFace(block_id) — 分流入口
//
// 度量系数（10个）：一律外推
// 坐标（3个）：
//   - 连接面 → fillGhostOnFaceFromVecs(..., block_id, 0) → assignGhostOnConnection
//   - 物理面 → fillGhostOnFaceFromVecs(...)               → assignGhostOnFace
// ====================================================================
PetscErrorCode JacGhostExtentBC::fillGhostCellOnFace(Mesh* mesh, int face, int block_id,
                                                      MultiBlockMesh* all_blocks)
{
    PetscErrorCode ierr;

    // ---- 度量系数（10个）：一律物理外推 ----
    ierr = fillGhostOnFaceFromVecs(mesh, face,
                                   mesh->getLocalMetricVecs(),
                                   all_blocks);
    CHKERRQ(ierr);

    // ---- 坐标（3个）：区分连接面 / 物理面 ----
    bool is_conn = false;
    if (all_blocks) {
        auto conn = all_blocks->findConnectedFace(block_id, face);
        is_conn = (conn.first >= 0);
    }

    if (is_conn) {
        ierr = fillGhostOnFaceFromVecs(mesh, face,
                                       mesh->getLocalCoordinateVecs(),
                                       all_blocks,
                                       block_id,  // 抄数模式
                                       0);         // comp_offset
    } else {
        ierr = fillGhostOnFaceFromVecs(mesh, face,
                                       mesh->getLocalCoordinateVecs(),
                                       all_blocks);
    }
    CHKERRQ(ierr);

    return 0;
}

// ====================================================================
// assignGhostOnConnection — 从对方块连接面抄坐标
// ====================================================================
PetscErrorCode JacGhostExtentBC::assignGhostOnConnection(
    Mesh* mesh, int block_id, int face,
    PetscReal*** arr, const DMDALocalInfo& info,
    MultiBlockMesh* all_blocks, int comp_idx)
{
    if (!all_blocks || comp_idx < 0 || comp_idx > 2) return 0;

        // ===== 块间抄数逻辑=====
        PetscInt nxg = mesh->getNxGlobal();
        PetscInt nyg = mesh->getNyGlobal();
        PetscInt nzg = mesh->getNzGlobal();

    int i_min, i_max, j_min, j_max, k_min, k_max;
    GetGhostRange(face, lap_, nxg, nyg, nzg, info,
                  i_min, i_max, j_min, j_max, k_min, k_max);
    if (i_min > i_max || j_min > j_max || k_min > k_max) return 0;

    const FaceConnectivity* fc = nullptr;
    bool is_block_a = true;
    for (auto& c : all_blocks->getFaceConnections()) {
        if (c.block_a == block_id && c.face_a == face) { fc = &c; is_block_a = true; break; }
        if (c.block_b == block_id && c.face_b == face) { fc = &c; is_block_a = false; break; }
    }
    if (!fc) return 0;

    const DonorSlab* slab = all_blocks->getDonorSlab(block_id, face);
    if (!slab || !slab->valid) return 0;

    const PetscReal* d_data = nullptr;
    if      (comp_idx == 0) d_data = slab->x.data();
    else if (comp_idx == 1) d_data = slab->y.data();
    else                    d_data = slab->z.data();
    if (!d_data) return 0;

    PetscInt s_i0 = slab->i0, s_j0 = slab->j0, s_k0 = slab->k0;
    PetscInt s_ni = slab->ni, s_nj = slab->nj, s_nk = slab->nk;

    int tr[3];
    if (is_block_a) {
        tr[0]=fc->transform[0]; tr[1]=fc->transform[1]; tr[2]=fc->transform[2];
    } else {
        for (int d=0; d<3; ++d) {
            tr[d] = 0;
            for (int s=0; s<3; ++s) {
                int av = fc->transform[s];
                if (av == 0) continue;
                int absv = (av > 0) ? av : -av;
                if (absv == d+1) { tr[d] = (av > 0 ? 1 : -1) * (s+1); break; }
            }
        }
    }

    int g_norm_axis = face / 2;
    int g_tan1 = (g_norm_axis == 0) ? 1 : 0;
    int g_tan2 = (g_norm_axis == 2) ? 1 : 2;

    const cgsize_t* pr = is_block_a ? fc->pointrange : fc->donorrange;
    PetscInt g_lo[3], g_hi[3];
    for (int a=0; a<3; ++a) {
        g_lo[a] = (PetscInt)pr[a*2];
        g_hi[a] = (PetscInt)pr[a*2+1];
        if (g_lo[a] > g_hi[a]) { auto t=g_lo[a]; g_lo[a]=g_hi[a]; g_hi[a]=t; }
    }

    int donor_face = is_block_a ? fc->face_b : fc->face_a;
    int d_norm_axis = donor_face / 2;

    const cgsize_t* dR = is_block_a ? fc->donorrange : fc->pointrange;
    PetscInt d_lo[3], d_hi[3];
    for (int a=0; a<3; ++a) {
        d_lo[a] = (PetscInt)dR[a*2];
        d_hi[a] = (PetscInt)dR[a*2+1];
        if (d_lo[a] > d_hi[a]) { auto t=d_lo[a]; d_lo[a]=d_hi[a]; d_hi[a]=t; }
    }

    PetscInt g_idx[3];
    for (PetscInt k = k_min; k <= k_max; ++k) {
        for (PetscInt j = j_min; j <= j_max; ++j) {
            for (PetscInt i = i_min; i <= i_max; ++i) {
                if (i >= 0 && i < nxg && j >= 0 && j < nyg && k >= 0 && k < nzg)
                    continue;

                g_idx[0]=i; g_idx[1]=j; g_idx[2]=k;

                PetscInt dist;
                if (face % 2 == 0)
                    dist = -g_idx[g_norm_axis];
                else
                    dist = g_idx[g_norm_axis] - (g_norm_axis==0 ? nxg-1 :
                                                  g_norm_axis==1 ? nyg-1 : nzg-1);
                if (dist < 1 || dist > lap_) continue;

                PetscInt o1 = g_idx[g_tan1] - g_lo[g_tan1];
                PetscInt o2 = g_idx[g_tan2] - g_lo[g_tan2];
                if (o1 < 0 || o1 > g_hi[g_tan1]-g_lo[g_tan1] ||
                    o2 < 0 || o2 > g_hi[g_tan2]-g_lo[g_tan2]) continue;

                PetscInt g_off[3] = {0,0,0};
                g_off[g_norm_axis] = dist;
                g_off[g_tan1] = o1;
                g_off[g_tan2] = o2;

                PetscInt d_idx[3] = {0,0,0};
                bool valid = true;
                for (int s=0; s<3; ++s) {
                    int dav = tr[s];
                    if (dav == 0) { valid=false; break; }
                    int dax = (dav>0 ? dav : -dav) - 1;
                    if (dax == d_norm_axis) {
                        if (donor_face % 2 == 0)
                            d_idx[dax] = d_lo[dax] + g_off[s];
                        else
                            d_idx[dax] = d_hi[dax] - g_off[s];
                    } else {
                        if (dav > 0)
                            d_idx[dax] = d_lo[dax] + g_off[s];
                        else
                            d_idx[dax] = d_hi[dax] - g_off[s];
                    }
                }
                if (!valid) continue;

                PetscInt si = d_idx[0] - s_i0;
                PetscInt sj = d_idx[1] - s_j0;
                PetscInt sk = d_idx[2] - s_k0;
                if (si < 0 || si >= s_ni || sj < 0 || sj >= s_nj ||
                    sk < 0 || sk >= s_nk) continue;

                arr[k][j][i] = d_data[(sk * s_nj + sj) * s_ni + si];
            }
        }
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

    // ★ 改为平坦化一维数组，避免 PetscInt(*)[3] 的 delete[] 类型问题
    PetscInt *all_sz  = nullptr;
    PetscInt *all_off = nullptr;
    if (rank == 0) {
        all_sz  = new PetscInt[nprocs * 3];
        all_off = new PetscInt[nprocs * 3];
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
                // 防止除零错误
                buf[p++] = (ajac[k][j][i] != 0.0) ? akx[k][j][i] / ajac[k][j][i] : 0.0;
            }

    int *recvcounts = nullptr, *displs = nullptr;
    PetscReal *all_buf = nullptr;
    if (rank == 0) {
        recvcounts = new int[nprocs];
        displs     = new int[nprocs];
        int total = 0;
        for (int r = 0; r < nprocs; ++r) {
            // all_sz[r*3+0] = gxm, all_sz[r*3+1] = gym, all_sz[r*3+2] = gzm
            recvcounts[r] = all_sz[r*3+0] * all_sz[r*3+1] * all_sz[r*3+2] * NVAR;
            displs[r] = total;
            total += recvcounts[r];
        }
        all_buf = new PetscReal[total];
    }
    MPI_Gatherv(buf, nlocal * NVAR, MPI_DOUBLE, all_buf, recvcounts, displs, MPI_DOUBLE, 0, comm);

    // rank 0 写入文件
    if (rank == 0) {
        PetscReal *grid = new PetscReal[GI * GJ * GK * NVAR]();
        for (int r = 0; r < nprocs; ++r) {
            PetscInt sz_i = all_sz[r*3+0];
            PetscInt sz_j = all_sz[r*3+1];
            PetscInt sz_k = all_sz[r*3+2];
            PetscInt off_i = all_off[r*3+0];
            PetscInt off_j = all_off[r*3+1];
            PetscInt off_k = all_off[r*3+2];
            PetscReal *src = all_buf + displs[r];
            PetscInt idx = 0;
            for (PetscInt kl = 0; kl < sz_k; ++kl) {
                PetscInt k = off_k + kl;
                for (PetscInt jl = 0; jl < sz_j; ++jl) {
                    PetscInt j = off_j + jl;
                    for (PetscInt il = 0; il < sz_i; ++il) {
                        PetscInt i = off_i + il;
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
    }

    // ★ 统一清理：只在 rank == 0 时分配过的才 delete
    if (rank == 0) {
        delete[] recvcounts;
        delete[] displs;
        delete[] all_buf;
    }

    delete[] buf;
    ierr = mesh->restoreLocalMetricArrays(akx, d1, d2, d3, d4, d5, d6, d7, d8, ajac); CHKERRQ(ierr);
    ierr = mesh->restoreLocalCoordinateArrays(axx, ayy, azz); CHKERRQ(ierr);

    delete[] all_sz;
    delete[] all_off;

    return 0;
}
