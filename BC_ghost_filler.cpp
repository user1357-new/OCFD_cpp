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
// setupMetrics — 三阶段度量系数初始化（格心版）
//
// Phase 1:  填格心坐标 face ghost（块间抄数 / BC 外推）
//           + 填格心坐标 edge/corner ghost
// Phase 2:  用已填 ghost 的格心坐标计算格心度量系数
// Phase 3:  填格心度量系数 face ghost + edge/corner ghost
//
// 注：格点→格心 8 点平均已在 Initialize() 中完成，此处从格心开始。
// ====================================================================
void MultiBlockMesh::setupMetrics(JacGhostExtentBC& filler,
                                   int edge_gtype, int metric_gtype)
{
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    PetscInt num_blocks = getNumBlocks();

    for (PetscInt b = 0; b < num_blocks; ++b) {
        Mesh* blk = getBlock(b);
        if (!blk) continue;
        for (int face = 0; face < 6; ++face) {
            auto conn = findConnectedFace((int)b, face);
            if (conn.first >= 0)
                filler.fillGhostOnFaceFromVecs(blk, face,
                    blk->getLocalCoordinateVecs(), this, (int)b, 0);
            else
                filler.fillGhostOnFaceFromVecs(blk, face,
                    blk->getLocalCoordinateVecs(), this);
        }
        blk->fillCellCoordGhost_CoordsOnly(edge_gtype);
    }
    if (rank == 0) PetscPrintf(PETSC_COMM_WORLD, "Phase 1 done.\n");

    for (PetscInt b = 0; b < num_blocks; ++b) {
        Mesh* blk = getBlock(b);
        if (!blk) continue;
        blk->computeCellJacobianFromFilledGhosts();
    }
    if (rank == 0) PetscPrintf(PETSC_COMM_WORLD, "Phase 2 done.\n");

    gatherFullMetrics();
    exchangeDonorSlabsMetricsPeriodic();

    for (PetscInt b = 0; b < num_blocks; ++b) {
        Mesh* blk = getBlock(b);
        if (!blk) continue;
        for (int face = 0; face < 6; ++face) {
            auto conn = findConnectedFace((int)b, face);
            bool is_periodic_conn = false;
            if (conn.first >= 0) {
                for (const auto& fc : face_connections_)
                    if (fc.is_periodic &&
                        ((fc.block_a == (int)b && fc.face_a == face) ||
                         (fc.block_b == (int)b && fc.face_b == face)))
                    { is_periodic_conn = true; break; }
            }
            if (conn.first >= 0 && is_periodic_conn)
                filler.fillGhostOnFaceFromVecs(blk, face,
                    blk->getLocalMetricVecs(), this, (int)b, 8);
            else
                filler.fillGhostOnFaceFromVecs(blk, face,
                    blk->getLocalMetricVecs(), this);
        }
        blk->fillCellCoordGhost_MetricsOnly(metric_gtype);
    }
    if (rank == 0) PetscPrintf(PETSC_COMM_WORLD, "Phase 3 done.\n");
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

    DM da_vec;
    ierr = VecGetDM(localVecs[0], &da_vec); CHKERRQ(ierr);
    if (!da_vec) {
        da_vec = mesh->getDM();
    }


    DMDALocalInfo info;
    ierr = DMDAGetLocalInfo(da_vec, &info); CHKERRQ(ierr);

    for (size_t n = 0; n < localVecs.size(); ++n) {
        if (localVecs[n] == nullptr) continue;
        PetscReal*** arr = nullptr;
        ierr = DMDAVecGetArray(da_vec, localVecs[n], &arr); CHKERRQ(ierr);

        if (block_id < 0) {
            ierr = assignGhostOnFace(mesh, face, arr, info, all_blocks);
        } else {
            ierr = assignGhostOnConnection(mesh, block_id, face, arr, info,
                                           all_blocks,
                                           comp_offset + static_cast<int>(n));
        }
        CHKERRQ(ierr);
        ierr = DMDAVecRestoreArray(da_vec, localVecs[n], &arr); CHKERRQ(ierr);
    }
    return 0;
}
// ====================================================================
// 块间赋值基类（默认实现为空）
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
// 遍历所有6个面
// ====================================================================
JacGhostExtentBC::JacGhostExtentBC(PetscInt gtype, PetscInt lap)
    : GhostCellFiller(lap)
{
    for (int f = 0; f < 6; ++f)
        ghost_cell_face_[f] = gtype;
}

// ====================================================================
// 具体的外推实现算法
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

    // ★ 从 info 反推物理域全局尺寸（适配格点和格心两种 DM）
    PetscInt nxg = info.mx;
    PetscInt nyg = info.my;
    PetscInt nzg = info.mz;

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
// assignGhostOnConnection — 从对方块连接面抄坐标（含 CGNS Transform 支持）
//
// 法向映射：通过 Transform 确定 donor 法向方向和面类型（MIN/MAX），
//          按 receiver 面类型 × donor 面类型四种组合选取公式：
//            MIN↔MIN:  -n-1            MIN↔MAX:  LAP+n
//            MAX↔MIN:  n-N             MAX↔MAX:  LAP-1-(n-N)
//          法向 slab 索引写入 slab_idx[donor_normal] 而非 slab_idx[receiver_normal]。
// 切向映射：根据 CGNS Transform 数组确定接收块 i/j/k → donor 块方向的对应关系，
//          支持方向交换（如 j↔k）和方向翻转（如 j→-j）。
// ====================================================================
PetscErrorCode JacGhostExtentBC::assignGhostOnConnection(
    Mesh* mesh, int block_id, int face,
    PetscReal*** arr, const DMDALocalInfo& info,
    MultiBlockMesh* all_blocks, int comp_idx)
{
    PetscErrorCode ierr;
    if (!all_blocks) return 0;

    const DonorSlab* slab = all_blocks->getDonorSlab(block_id, face);
    if (!slab || !slab->valid) return 0;

    int i_min, i_max, j_min, j_max, k_min, k_max;
    PetscInt NX = mesh->getNxGlobal();
    PetscInt NY = mesh->getNyGlobal();
    PetscInt NZ = mesh->getNzGlobal();
    PetscInt LAP = getLap();

    GetGhostRange(face, LAP, NX, NY, NZ, info, i_min, i_max, j_min, j_max, k_min, k_max);

    // ---- 查找 FaceConnectivity，获取 CGNS Transform ----
    const auto& connections = all_blocks->getFaceConnections();
    const FaceConnectivity* fc = nullptr;
    for (const auto& c : connections) {
        if ((c.block_a == block_id && c.face_a == face) ||
            (c.block_b == block_id && c.face_b == face)) {
            fc = &c;
            break;
        }
    }

    // transR2D[r] = {donor_dir, sign}: 接收块方向 r (0=i,1=j,2=k) → donor 方向及符号
    int transR2D[3][2];
    for (int r = 0; r < 3; r++) {
        transR2D[r][0] = r;   // 默认: identity
        transR2D[r][1] = 1;
    }

    if (fc) {
        if (block_id == fc->block_a) {
            // fc->transform 就是 block_a(接收) → block_b(donor)
            for (int r = 0; r < 3; r++) {
                int v = fc->transform[r];
                transR2D[r][0] = (v < 0 ? -v : v) - 1;
                transR2D[r][1] = (v > 0) ? 1 : -1;
            }
        } else {
            // 接收 = block_b, donor = block_a: 需要 fc->transform 的逆
            for (int r = 0; r < 3; r++) {
                for (int t = 0; t < 3; t++) {
                    int at = fc->transform[t];
                    if ((at < 0 ? -at : at) == r + 1) {
                        transR2D[r][0] = t;
                        transR2D[r][1] = (fc->transform[t] > 0) ? 1 : -1;
                        break;
                    }
                }
            }
        }
    }

    PetscInt r_normal = face / 2;  // 接收块的面法向: 0=i, 1=j, 2=k

    // ★ 通过 Transform 确定 donor 侧的法向方向
    int donor_normal = transR2D[r_normal][0];   // donor 法向: 0=i, 1=j, 2=k
    int normal_sign  = transR2D[r_normal][1];   // >0→同向, <0→反向 (用于切向映射)
    bool recv_is_min = (face % 2 == 0);          // MIN面: 0,2,4; MAX面: 1,3,5

    // ★ donor_is_min 必须从 detectFace 产出的实际 donor face 确定，
    //    不能从 transform 符号推导。因为对于 MIN↔MIN 连接（两个块都是
    //    i=1 MIN 面但 transform sign=-1），ghost→slab 正向映射才是正确的。
    int donor_face;
    if (fc) {
        // ★ 必须同时匹配 block 和 face，自周期性时 block_a==block_b
        if (block_id == fc->block_a && face == fc->face_a)
            donor_face = fc->face_b;
        else
            donor_face = fc->face_a;
    } else {
        donor_face = (normal_sign > 0) ? face : (face ^ 1);
    }
    bool donor_is_min = (donor_face % 2 == 0);   // MIN面: 0,2,4; MAX面: 1,3,5

    // ★ 提取接收块自身的连接面范围 (0-based 格心索引)
    //    pointrange / donorrange 格式: [imin, imax, jmin, jmax, kmin, kmax]
    const cgsize_t* recv_range = nullptr;
    if (fc) {
        if (block_id == fc->block_a)
            recv_range = fc->pointrange;
        else
            recv_range = fc->donorrange;
    }

    for (PetscInt k = k_min; k <= k_max; k++) {
        for (PetscInt j = j_min; j <= j_max; j++) {
            for (PetscInt i = i_min; i <= i_max; i++) {
                if (i >= 0 && i < NX && j >= 0 && j < NY && k >= 0 && k < NZ)
                    continue;

                // slab_idx[0,1,2] = {si, sj, sk}
                PetscInt slab_idx[3];

                // ---- 法向：根据 receiver + donor 面类型四选一，写入 donor 法向索引 ----
                PetscInt r_val_n = (r_normal == 0) ? i : (r_normal == 1) ? j : k;
                PetscInt N_norm  = (r_normal == 0) ? NX : (r_normal == 1) ? NY : NZ;

                if (recv_is_min && donor_is_min)
                    slab_idx[donor_normal] = -r_val_n - 1;
                else if (recv_is_min && !donor_is_min)
                    slab_idx[donor_normal] = LAP + r_val_n;
                else if (!recv_is_min && donor_is_min)
                    slab_idx[donor_normal] = r_val_n - N_norm;
                else  // !recv_is_min && !donor_is_min
                    slab_idx[donor_normal] = LAP - 1 - (r_val_n - N_norm);

                // ---- 切向：用 Transform 确定映射 ----
                for (int r = 0; r < 3; r++) {
                    if (r == r_normal) continue;  // 法向已处理

                    int d_donor = transR2D[r][0];  // 该接收方向映射到 donor 的哪个方向
                    int sign    = transR2D[r][1];

                    PetscInt r_val = (r == 0) ? i : (r == 1) ? j : k;

                    // 用接收块自己的连接起始索引算局部偏移
                    PetscInt recv_s0;
                    if (r == 0)      recv_s0 = recv_range[0];
                    else if (r == 1) recv_s0 = recv_range[2];
                    else             recv_s0 = recv_range[4];

                    PetscInt local_off = r_val - recv_s0;  // 连接区域内的相对位置

                    PetscInt sn;
                    if (d_donor == 0)      sn = slab->ni;
                    else if (d_donor == 1) sn = slab->nj;
                    else                   sn = slab->nk;

                    // 直接映射到 donor slab 的局部索引（slab 按连接区域精确打包，尺寸一致）
                    if (sign > 0)
                        slab_idx[d_donor] = local_off;
                    else
                        slab_idx[d_donor] = sn - 1 - local_off;  // 方向翻转
                }

                PetscInt si = slab_idx[0], sj = slab_idx[1], sk = slab_idx[2];

                if (si < 0 || si >= slab->ni || sj < 0 || sj >= slab->nj || sk < 0 || sk >= slab->nk)
                    continue;

                PetscInt idx = (sk * slab->nj + sj) * slab->ni + si;
                const PetscReal* d_data = nullptr;
                if (comp_idx < 3) {
                    if      (comp_idx == 0) d_data = slab->x.data();
                    else if (comp_idx == 1) d_data = slab->y.data();
                    else                    d_data = slab->z.data();
                } else if (comp_idx < 8) {
                    d_data = slab->u[comp_idx - 3].data();
                } else if (comp_idx < 18) {
                    d_data = slab->m[comp_idx - 8].data();
                }
                if (d_data) {
                    PetscReal val = d_data[idx];
                    // ★ 周期连接 + 坐标分量 → 施加平移
                    if (fc && fc->is_periodic && comp_idx < 3) {
                        if (recv_is_min)
                            val -= fc->translate[comp_idx];
                        else
                            val += fc->translate[comp_idx];
                    }
                    arr[k][j][i] = val;
                }
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

    MPI_Comm comm = mesh->getComm();
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    const int NVAR = 3;  // 只导出格心坐标 X,Y,Z（含 ghost）

    PetscInt LAP = mesh->getLAP();
    PetscInt nxg = mesh->getNxGlobal();
    PetscInt nyg = mesh->getNyGlobal();
    PetscInt nzg = mesh->getNzGlobal();
    PetscInt GI = nxg + 2 * LAP;
    PetscInt GJ = nyg + 2 * LAP;
    PetscInt GK = nzg + 2 * LAP;

    // 直接访问格点 local Vec
    DM da = mesh->getDM();
    DMDALocalInfo info;
    ierr = DMDAGetLocalInfo(da, &info); CHKERRQ(ierr);

    PetscReal ***axx, ***ayy, ***azz;
    Vec axx_v = mesh->getLocalCoordinateVecs()[0];
    Vec ayy_v = mesh->getLocalCoordinateVecs()[1];
    Vec azz_v = mesh->getLocalCoordinateVecs()[2];
    ierr = DMDAVecGetArray(da, axx_v, &axx); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, ayy_v, &ayy); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, azz_v, &azz); CHKERRQ(ierr);

    PetscInt send_sz[3] = {info.gxm, info.gym, info.gzm};
    PetscInt send_off[3] = {info.gxs, info.gys, info.gzs};

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
            }

    int *recvcounts = nullptr, *displs = nullptr;
    PetscReal *all_buf = nullptr;
    if (rank == 0) {
        recvcounts = new int[nprocs];
        displs     = new int[nprocs];
        int total = 0;
        for (int r = 0; r < nprocs; ++r) {
            recvcounts[r] = all_sz[r*3+0] * all_sz[r*3+1] * all_sz[r*3+2] * NVAR;
            displs[r] = total;
            total += recvcounts[r];
        }
        all_buf = new PetscReal[total];
    }
    MPI_Gatherv(buf, nlocal * NVAR, MPI_DOUBLE, all_buf, recvcounts, displs, MPI_DOUBLE, 0, comm);

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
            << "VARIABLES=\"X\",\"Y\",\"Z\"\n"
            << "ZONE T=\"" << base_filename << "\",I=" << GI
            << ",J=" << GJ << ",K=" << GK << ",F=POINT\n";
        out.precision(8);
        out << std::scientific;
        PetscInt ntot = GI * GJ * GK;
        for (PetscInt n = 0; n < ntot; ++n)
            out << grid[n * NVAR]     << " "
                << grid[n * NVAR + 1] << " "
                << grid[n * NVAR + 2] << "\n";
        out.close();
        std::cout << "Exported: " << fname << " (" << ntot << " pts)\n";

        delete[] grid;
    }

    if (rank == 0) {
        delete[] recvcounts;
        delete[] displs;
        delete[] all_buf;
    }
    delete[] buf;

    ierr = DMDAVecRestoreArray(da, axx_v, &axx); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, ayy_v, &ayy); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, azz_v, &azz); CHKERRQ(ierr);

    delete[] all_sz;
    delete[] all_off;

    return 0;
}
// ====================================================================
// exchangeDonorSlabs — 为每个连接面交换 LAP 层 slab
// ====================================================================
void MultiBlockMesh::exchangeDonorSlabs(const std::vector<PetscInt>& block_start_rank)
{
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    PetscInt num_blocks = (PetscInt)full_coords_.size();

    PetscInt my_block = -1;
    for (PetscInt b = 0; b < num_blocks; ++b) {
        if (full_coords_[b].valid) { my_block = b; break; }
    }

    // ============================
    // ★ packSlab: 法向从 interior layer 1 开始打包
    // ============================
               auto packSlab = [&](PetscInt donor_block, PetscInt donor_face,
                            const cgsize_t* donorrange) -> DonorSlab
        {
            DonorSlab slab;
            const auto& coords = full_coords_[donor_block];
            if (!coords.valid) return slab;

            PetscInt NI = coords.ni, NJ = coords.nj, NK = coords.nk;

            PetscInt dr_imin = (PetscInt)donorrange[0];
            PetscInt dr_imax = (PetscInt)donorrange[1];
            PetscInt dr_jmin = (PetscInt)donorrange[2];
            PetscInt dr_jmax = (PetscInt)donorrange[3];
            PetscInt dr_kmin = (PetscInt)donorrange[4];
            PetscInt dr_kmax = (PetscInt)donorrange[5];

            // ★ 防御性 clamp（detectFace 已做顶点→格心转换，正常不应触发）
            if (dr_imin < 0)  dr_imin = 0;
            if (dr_jmin < 0)  dr_jmin = 0;
            if (dr_kmin < 0)  dr_kmin = 0;
            if (dr_imax >= NI) dr_imax = NI - 1;
            if (dr_jmax >= NJ) dr_jmax = NJ - 1;
            if (dr_kmax >= NK) dr_kmax = NK - 1;

            if (dr_imin > dr_imax || dr_jmin > dr_jmax || dr_kmin > dr_kmax)
                return slab;

            switch (donor_face) {
            case 0: slab.i0 = 0;              slab.j0 = dr_jmin;  slab.k0 = dr_kmin;
                    slab.ni = LAP;            slab.nj = dr_jmax - dr_jmin + 1;
                                              slab.nk = dr_kmax - dr_kmin + 1; break;
            case 1: slab.i0 = NI - LAP;       slab.j0 = dr_jmin;  slab.k0 = dr_kmin;
                    slab.ni = LAP;            slab.nj = dr_jmax - dr_jmin + 1;
                                              slab.nk = dr_kmax - dr_kmin + 1; break;
            case 2: slab.i0 = dr_imin;        slab.j0 = 0;        slab.k0 = dr_kmin;
                    slab.ni = dr_imax - dr_imin + 1; slab.nj = LAP;
                                              slab.nk = dr_kmax - dr_kmin + 1; break;
            case 3: slab.i0 = dr_imin;        slab.j0 = NJ - LAP; slab.k0 = dr_kmin;
                    slab.ni = dr_imax - dr_imin + 1; slab.nj = LAP;
                                              slab.nk = dr_kmax - dr_kmin + 1; break;
            case 4: slab.i0 = dr_imin;        slab.j0 = dr_jmin;  slab.k0 = 0;
                    slab.ni = dr_imax - dr_imin + 1; slab.nj = dr_jmax - dr_jmin + 1;
                                              slab.nk = LAP; break;
            case 5: slab.i0 = dr_imin;        slab.j0 = dr_jmin;  slab.k0 = NK - LAP;
                    slab.ni = dr_imax - dr_imin + 1; slab.nj = dr_jmax - dr_jmin + 1;
                                              slab.nk = LAP; break;
            default: return slab;
            }

            PetscInt total = slab.ni * slab.nj * slab.nk;
            slab.x.resize((size_t)total);
            slab.y.resize((size_t)total);
            slab.z.resize((size_t)total);

            PetscInt idx = 0;
            for (PetscInt k = 0; k < slab.nk; ++k) {
                PetscInt kg = slab.k0 + k;
                for (PetscInt j = 0; j < slab.nj; ++j) {
                    PetscInt jg = slab.j0 + j;
                    for (PetscInt i = 0; i < slab.ni; ++i) {
                        PetscInt ig = slab.i0 + i;
                        PetscInt gi = (kg * NJ + jg) * NI + ig;
                        slab.x[idx] = coords.x[gi];
                        slab.y[idx] = coords.y[gi];
                        slab.z[idx] = coords.z[gi];
                        ++idx;
                    }
                }
            }
            slab.valid = true;
            return slab;
        };



    // ======== 世界广播 ========
    PetscInt nConn = (PetscInt)face_connections_.size();
    for (PetscInt ci = 0; ci < nConn; ++ci) {
        auto& conn = face_connections_[ci];

        // ── 方向1: block_a ← block_b ──
        {
            PetscMPIInt root = (PetscMPIInt)block_start_rank[conn.block_b];
            PetscMPIInt dims[6];
            PetscMPIInt total = 0;
            std::vector<PetscReal> xb, yb, zb;

            if (rank == root) {
                DonorSlab s = packSlab((PetscInt)conn.block_b,
                                       (PetscInt)conn.face_b,
                                       conn.donorrange);
                if (s.valid) {
                    dims[0] = (PetscMPIInt)s.ni; dims[1] = (PetscMPIInt)s.nj;
                    dims[2] = (PetscMPIInt)s.nk; dims[3] = (PetscMPIInt)s.i0;
                    dims[4] = (PetscMPIInt)s.j0; dims[5] = (PetscMPIInt)s.k0;
                    total   = (PetscMPIInt)(s.ni * s.nj * s.nk);
                    xb = std::move(s.x);
                    yb = std::move(s.y);
                    zb = std::move(s.z);
                }
            }
            MPI_Bcast(dims,  6, MPI_INT,    root, PETSC_COMM_WORLD);
            MPI_Bcast(&total, 1, MPI_INT,    root, PETSC_COMM_WORLD);

            // ★★★ 广播数据（之前缺失） ★★★
            if (total > 0) {
                if (rank != root) {
                    xb.resize((size_t)total);
                    yb.resize((size_t)total);
                    zb.resize((size_t)total);
                }
                MPI_Bcast(xb.data(), total, MPI_DOUBLE, root, PETSC_COMM_WORLD);
                MPI_Bcast(yb.data(), total, MPI_DOUBLE, root, PETSC_COMM_WORLD);
                MPI_Bcast(zb.data(), total, MPI_DOUBLE, root, PETSC_COMM_WORLD);
            }

            PetscMPIInt recv_root = (PetscMPIInt)block_start_rank[conn.block_a];
            if (rank == recv_root) {
                DonorSlab s;
                s.ni = (PetscInt)dims[0]; s.nj = (PetscInt)dims[1];
                s.nk = (PetscInt)dims[2]; s.i0 = (PetscInt)dims[3];
                s.j0 = (PetscInt)dims[4]; s.k0 = (PetscInt)dims[5];
                s.valid = true;
                if (total > 0) {
                    s.x = std::move(xb);
                    s.y = std::move(yb);
                    s.z = std::move(zb);
                }
                donor_slabs_[conn.block_a][conn.face_a] = std::move(s);
            }
        }

        // ── 方向2: block_b ← block_a ──
        {
            PetscMPIInt root = (PetscMPIInt)block_start_rank[conn.block_a];
            PetscMPIInt dims[6];
            PetscMPIInt total = 0;
            std::vector<PetscReal> xb, yb, zb;

            if (rank == root) {
                DonorSlab s = packSlab((PetscInt)conn.block_a,
                                       (PetscInt)conn.face_a,
                                       conn.pointrange);
                if (s.valid) {
                    dims[0] = (PetscMPIInt)s.ni; dims[1] = (PetscMPIInt)s.nj;
                    dims[2] = (PetscMPIInt)s.nk; dims[3] = (PetscMPIInt)s.i0;
                    dims[4] = (PetscMPIInt)s.j0; dims[5] = (PetscMPIInt)s.k0;
                    total   = (PetscMPIInt)(s.ni * s.nj * s.nk);
                    xb = std::move(s.x);
                    yb = std::move(s.y);
                    zb = std::move(s.z);
                }
            }
            MPI_Bcast(dims,  6, MPI_INT,    root, PETSC_COMM_WORLD);
            MPI_Bcast(&total, 1, MPI_INT,    root, PETSC_COMM_WORLD);

            // ★★★ 广播数据（之前缺失） ★★★
            if (total > 0) {
                if (rank != root) {
                    xb.resize((size_t)total);
                    yb.resize((size_t)total);
                    zb.resize((size_t)total);
                }
                MPI_Bcast(xb.data(), total, MPI_DOUBLE, root, PETSC_COMM_WORLD);
                MPI_Bcast(yb.data(), total, MPI_DOUBLE, root, PETSC_COMM_WORLD);
                MPI_Bcast(zb.data(), total, MPI_DOUBLE, root, PETSC_COMM_WORLD);
            }

            PetscMPIInt recv_root = (PetscMPIInt)block_start_rank[conn.block_b];
            if (rank == recv_root) {
                DonorSlab s;
                s.ni = (PetscInt)dims[0]; s.nj = (PetscInt)dims[1];
                s.nk = (PetscInt)dims[2]; s.i0 = (PetscInt)dims[3];
                s.j0 = (PetscInt)dims[4]; s.k0 = (PetscInt)dims[5];
                s.valid = true;
                if (total > 0) {
                    s.x = std::move(xb);
                    s.y = std::move(yb);
                    s.z = std::move(zb);
                }
                donor_slabs_[conn.block_b][conn.face_b] = std::move(s);
            }
        }
    }

    // ======== 各块内广播 ========
    for (PetscInt b = 0; b < num_blocks; ++b) {
        if ((PetscInt)my_block != b) continue;
        MPI_Comm comm = block_comms_[b];
        for (PetscMPIInt face = 0; face < 6; ++face) {
            DonorSlab& slab = donor_slabs_[b][face];
            PetscMPIInt vf = slab.valid ? 1 : 0;
            MPI_Bcast(&vf, 1, MPI_INT, 0, comm);
            slab.valid = (vf != 0);
            if (!slab.valid) continue;

            PetscMPIInt dims[6];
            PetscMPIInt total;
            if (rank == (PetscMPIInt)block_start_rank[b]) {
                dims[0] = (PetscMPIInt)slab.ni; dims[1] = (PetscMPIInt)slab.nj;
                dims[2] = (PetscMPIInt)slab.nk; dims[3] = (PetscMPIInt)slab.i0;
                dims[4] = (PetscMPIInt)slab.j0; dims[5] = (PetscMPIInt)slab.k0;
                total  = (PetscMPIInt)(slab.ni * slab.nj * slab.nk);
            }
            MPI_Bcast(dims, 6, MPI_INT, 0, comm);
            MPI_Bcast(&total, 1, MPI_INT, 0, comm);

            if (rank != (PetscMPIInt)block_start_rank[b]) {
                slab.ni = dims[0]; slab.nj = dims[1]; slab.nk = dims[2];
                slab.i0 = dims[3]; slab.j0 = dims[4]; slab.k0 = dims[5];
                if (total > 0) {
                    slab.x.resize((size_t)total);
                    slab.y.resize((size_t)total);
                    slab.z.resize((size_t)total);
                }
            }
            if (total > 0) {
                MPI_Bcast(slab.x.data(), total, MPI_DOUBLE, 0, comm);
                MPI_Bcast(slab.y.data(), total, MPI_DOUBLE, 0, comm);
                MPI_Bcast(slab.z.data(), total, MPI_DOUBLE, 0, comm);
            }
        }
    }
}
// ====================================================================
// getDonorSlab — 查询本地缓存的 donor slab
// ====================================================================
const DonorSlab* MultiBlockMesh::getDonorSlab(int block_id, int face) const
{
    if (block_id < 0 || block_id >= (int)donor_slabs_.size()) return nullptr;
    if (face < 0 || face > 5) return nullptr;
    if (!donor_slabs_[block_id][face].valid) return nullptr;
    return &donor_slabs_[block_id][face];
}

// ====================================================================
// gatherFullFlow — 从 DMDA 汇集各块完整守恒量到 full_coords_
// ====================================================================
void MultiBlockMesh::gatherFullFlow()
{
    PetscMPIInt world_rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &world_rank);
    PetscInt num_blocks = getNumBlocks();

    for (PetscInt b = 0; b < num_blocks; ++b) {
        Mesh* blk = getBlock(b);
        if (!blk) continue;

        MPI_Comm comm = block_comms_[b];
        int rank_in_block, size_in_block;
        MPI_Comm_rank(comm, &rank_in_block);
        MPI_Comm_size(comm, &size_in_block);

        PetscInt nxg = blk->getNxGlobal();
        PetscInt nyg = blk->getNyGlobal();
        PetscInt nzg = blk->getNzGlobal();
        PetscInt ntotal = nxg * nyg * nzg;

        DM da = blk->getDM();
        DMDALocalInfo info;
        DMDAGetLocalInfo(da, &info);

        PetscInt xs = info.xs, ys = info.ys, zs = info.zs;
        PetscInt xm = info.xm, ym = info.ym, zm = info.zm;
        PetscInt nlocal = xm * ym * zm;

        // 分配 full_coords_ 的 U 数组
        for (int comp = 0; comp < 5; ++comp)
            full_coords_[b].u[comp].resize((size_t)ntotal);

        for (int comp = 0; comp < 5; ++comp) {
            // ---- 读取本地 DMDA 切片 ----
            PetscReal*** arr;
            DMDAVecGetArray(da, blk->getURef(comp), &arr);

            std::vector<PetscReal> local_buf((size_t)nlocal);
            PetscInt idx = 0;
            for (PetscInt k = zs; k < zs + zm; ++k)
                for (PetscInt j = ys; j < ys + ym; ++j)
                    for (PetscInt i = xs; i < xs + xm; ++i)
                        local_buf[idx++] = arr[k][j][i];

            DMDAVecRestoreArray(da, blk->getURef(comp), &arr);

            // ---- 汇集元数据: 每个进程的 (xs,ys,zs,xm,ym,zm) ----
            PetscInt my_meta[6] = {xs, ys, zs, xm, ym, zm};
            std::vector<PetscInt> all_meta;
            if (rank_in_block == 0)
                all_meta.resize((size_t)size_in_block * 6);
            MPI_Gather(my_meta, 6, MPIU_INT,
                       all_meta.data(), 6, MPIU_INT, 0, comm);

            // ---- 汇集数据量 ----
            int my_n = (int)nlocal;
            std::vector<int> all_n;
            if (rank_in_block == 0)
                all_n.resize((size_t)size_in_block);
            MPI_Gather(&my_n, 1, MPI_INT,
                       all_n.data(), 1, MPI_INT, 0, comm);

            // ---- 汇集数据 ----
            std::vector<int> displs;
            std::vector<PetscReal> all_data;
            if (rank_in_block == 0) {
                int total = 0;
                displs.resize((size_t)size_in_block);
                for (int r = 0; r < size_in_block; ++r) {
                    displs[r] = total;
                    total += all_n[r];
                }
                all_data.resize((size_t)total);
            }
            MPI_Gatherv(local_buf.data(), my_n, MPI_DOUBLE,
                        all_data.data(), all_n.data(), displs.data(),
                        MPI_DOUBLE, 0, comm);

            // ---- rank 0 组装完整网格 ----
            if (rank_in_block == 0) {
                auto& fu = full_coords_[b].u[comp];
                for (int r = 0; r < size_in_block; ++r) {
                    PetscInt* m = &all_meta[(size_t)r * 6];
                    PetscInt r_xs = m[0], r_ys = m[1], r_zs = m[2];
                    PetscInt r_xm = m[3], r_ym = m[4], r_zm = m[5];
                    PetscReal* src = all_data.data() + displs[r];
                    PetscInt src_idx = 0;
                    for (PetscInt k = r_zs; k < r_zs + r_zm; ++k)
                        for (PetscInt j = r_ys; j < r_ys + r_ym; ++j)
                            for (PetscInt i = r_xs; i < r_xs + r_xm; ++i)
                                fu[(size_t)(k * nyg + j) * nxg + i] = src[src_idx++];
                }
            }

            // ---- 广播完整网格到块内所有进程 ----
            MPI_Bcast(full_coords_[b].u[comp].data(), (int)ntotal,
                      MPI_DOUBLE, 0, comm);
        }

        full_coords_[b].flow_ready = true;
    }

    PetscPrintf(PETSC_COMM_WORLD, "[gatherFullFlow] done.\n");
}

// ====================================================================
// exchangeDonorSlabsU — 交换守恒量 donor slab
// 复用 exchangeDonorSlabs 已建立的 slab 几何信息，只传 U 数据
// ====================================================================
void MultiBlockMesh::exchangeDonorSlabsU()
{
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    PetscInt nConn = (PetscInt)face_connections_.size();
    if (nConn == 0) return;

    for (PetscInt ci = 0; ci < nConn; ++ci) {
        auto& conn = face_connections_[ci];

        // ── 方向1: block_a ← block_b ──
        {
            PetscMPIInt root = (PetscMPIInt)block_start_rank_[conn.block_b];
            PetscMPIInt dims[6];
            PetscMPIInt total = 0;
            std::vector<PetscReal> ub[5];

            if (rank == root) {
                const auto& coords = full_coords_[conn.block_b];
                if (coords.flow_ready) {
                    // ★ 从已知 donor_slabs 取几何，从 full_coords_ 取 U
                    const DonorSlab& ref = donor_slabs_[conn.block_b][conn.face_b];
                    if (ref.valid) {
                        dims[0] = (PetscMPIInt)ref.ni; dims[1] = (PetscMPIInt)ref.nj;
                        dims[2] = (PetscMPIInt)ref.nk; dims[3] = (PetscMPIInt)ref.i0;
                        dims[4] = (PetscMPIInt)ref.j0; dims[5] = (PetscMPIInt)ref.k0;
                        total = (PetscMPIInt)(ref.ni * ref.nj * ref.nk);
                        PetscInt NI = coords.ni, NJ = coords.nj;
                        // 从 full_coords_ 打包 U slab
                        for (int c = 0; c < 5; ++c) {
                            ub[c].resize((size_t)total);
                            PetscInt slot = 0;
                            for (PetscInt k = 0; k < ref.nk; ++k) {
                                PetscInt kg = ref.k0 + k;
                                for (PetscInt j = 0; j < ref.nj; ++j) {
                                    PetscInt jg = ref.j0 + j;
                                    for (PetscInt i = 0; i < ref.ni; ++i) {
                                        PetscInt ig = ref.i0 + i;
                                        PetscInt gi = ((size_t)kg * NJ + jg) * NI + ig;
                                        ub[c][slot++] = coords.u[c][gi];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            MPI_Bcast(dims,  6, MPI_INT,    root, PETSC_COMM_WORLD);
            MPI_Bcast(&total, 1, MPI_INT,    root, PETSC_COMM_WORLD);

            if (total > 0) {
                for (int c = 0; c < 5; ++c) {
                    if (rank != root) ub[c].resize((size_t)total);
                    MPI_Bcast(ub[c].data(), total, MPI_DOUBLE, root, PETSC_COMM_WORLD);
                }
            }

            PetscMPIInt recv_root = (PetscMPIInt)block_start_rank_[conn.block_a];
            if (rank == recv_root && total > 0) {
                auto& slab = donor_slabs_[conn.block_a][conn.face_a];
                for (int c = 0; c < 5; ++c)
                    slab.u[c] = std::move(ub[c]);
            }
        }

        // ── 方向2: block_b ← block_a ──
        {
            PetscMPIInt root = (PetscMPIInt)block_start_rank_[conn.block_a];
            PetscMPIInt dims[6];
            PetscMPIInt total = 0;
            std::vector<PetscReal> ub[5];

            if (rank == root) {
                const auto& coords = full_coords_[conn.block_a];
                if (coords.flow_ready) {
                    const DonorSlab& ref = donor_slabs_[conn.block_a][conn.face_a];
                    if (ref.valid) {
                        dims[0] = (PetscMPIInt)ref.ni; dims[1] = (PetscMPIInt)ref.nj;
                        dims[2] = (PetscMPIInt)ref.nk; dims[3] = (PetscMPIInt)ref.i0;
                        dims[4] = (PetscMPIInt)ref.j0; dims[5] = (PetscMPIInt)ref.k0;
                        total = (PetscMPIInt)(ref.ni * ref.nj * ref.nk);
                        PetscInt NI = coords.ni, NJ = coords.nj;
                        for (int c = 0; c < 5; ++c) {
                            ub[c].resize((size_t)total);
                            PetscInt slot = 0;
                            for (PetscInt k = 0; k < ref.nk; ++k) {
                                PetscInt kg = ref.k0 + k;
                                for (PetscInt j = 0; j < ref.nj; ++j) {
                                    PetscInt jg = ref.j0 + j;
                                    for (PetscInt i = 0; i < ref.ni; ++i) {
                                        PetscInt ig = ref.i0 + i;
                                        PetscInt gi = ((size_t)kg * NJ + jg) * NI + ig;
                                        ub[c][slot++] = coords.u[c][gi];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            MPI_Bcast(dims,  6, MPI_INT,    root, PETSC_COMM_WORLD);
            MPI_Bcast(&total, 1, MPI_INT,    root, PETSC_COMM_WORLD);

            if (total > 0) {
                for (int c = 0; c < 5; ++c) {
                    if (rank != root) ub[c].resize((size_t)total);
                    MPI_Bcast(ub[c].data(), total, MPI_DOUBLE, root, PETSC_COMM_WORLD);
                }
            }

            PetscMPIInt recv_root = (PetscMPIInt)block_start_rank_[conn.block_b];
            if (rank == recv_root && total > 0) {
                auto& slab = donor_slabs_[conn.block_b][conn.face_b];
                for (int c = 0; c < 5; ++c)
                    slab.u[c] = std::move(ub[c]);
            }
        }
    }

    // ======== 各块内广播 ========
    PetscInt num_blocks = getNumBlocks();
    PetscInt my_block = -1;
    for (PetscInt b = 0; b < num_blocks; ++b) {
        if (full_coords_[b].valid) { my_block = b; break; }
    }

    for (PetscInt b = 0; b < num_blocks; ++b) {
        if (my_block != b) continue;
        MPI_Comm comm = block_comms_[b];
        for (PetscMPIInt face = 0; face < 6; ++face) {
            DonorSlab& slab = donor_slabs_[b][face];
            if (!slab.valid) continue;

            PetscMPIInt total = (PetscMPIInt)(slab.ni * slab.nj * slab.nk);
            if (total <= 0) continue;

            // 非根进程分配空间
            if (rank != (PetscMPIInt)block_start_rank_[b]) {
                for (int c = 0; c < 5; ++c)
                    slab.u[c].resize((size_t)total);
            }

            for (int c = 0; c < 5; ++c)
                MPI_Bcast(slab.u[c].data(), total, MPI_DOUBLE, 0, comm);
        }
    }

    if (rank == 0)
        PetscPrintf(PETSC_COMM_WORLD, "[exchangeDonorSlabsU] done.\n");
}

// ====================================================================
// exchangeDonorSlabsPeriodic — 为周期性连接打包坐标 DonorSlab
// ====================================================================
void MultiBlockMesh::exchangeDonorSlabsPeriodic()
{
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    PetscInt num_blocks = (PetscInt)full_coords_.size();

    PetscInt my_block = -1;
    for (PetscInt b = 0; b < num_blocks; ++b) {
        if (full_coords_[b].valid) { my_block = b; break; }
    }

    auto packSlab = [&](PetscInt donor_block, PetscInt donor_face,
                        const cgsize_t* donorrange) -> DonorSlab
    {
        DonorSlab slab;
        const auto& coords = full_coords_[donor_block];
        if (!coords.valid) return slab;

        PetscInt NI = coords.ni, NJ = coords.nj, NK = coords.nk;

        PetscInt dr_imin = (PetscInt)donorrange[0];
        PetscInt dr_imax = (PetscInt)donorrange[1];
        PetscInt dr_jmin = (PetscInt)donorrange[2];
        PetscInt dr_jmax = (PetscInt)donorrange[3];
        PetscInt dr_kmin = (PetscInt)donorrange[4];
        PetscInt dr_kmax = (PetscInt)donorrange[5];

        if (dr_imin < 0)  dr_imin = 0;
        if (dr_jmin < 0)  dr_jmin = 0;
        if (dr_kmin < 0)  dr_kmin = 0;
        if (dr_imax >= NI) dr_imax = NI - 1;
        if (dr_jmax >= NJ) dr_jmax = NJ - 1;
        if (dr_kmax >= NK) dr_kmax = NK - 1;

        if (dr_imin > dr_imax || dr_jmin > dr_jmax || dr_kmin > dr_kmax)
            return slab;

        switch (donor_face) {
        case 0: slab.i0 = 0;              slab.j0 = dr_jmin;  slab.k0 = dr_kmin;
                slab.ni = LAP;            slab.nj = dr_jmax - dr_jmin + 1;
                                          slab.nk = dr_kmax - dr_kmin + 1; break;
        case 1: slab.i0 = NI - LAP;       slab.j0 = dr_jmin;  slab.k0 = dr_kmin;
                slab.ni = LAP;            slab.nj = dr_jmax - dr_jmin + 1;
                                          slab.nk = dr_kmax - dr_kmin + 1; break;
        case 2: slab.i0 = dr_imin;        slab.j0 = 0;        slab.k0 = dr_kmin;
                slab.ni = dr_imax - dr_imin + 1; slab.nj = LAP;
                                          slab.nk = dr_kmax - dr_kmin + 1; break;
        case 3: slab.i0 = dr_imin;        slab.j0 = NJ - LAP; slab.k0 = dr_kmin;
                slab.ni = dr_imax - dr_imin + 1; slab.nj = LAP;
                                          slab.nk = dr_kmax - dr_kmin + 1; break;
        case 4: slab.i0 = dr_imin;        slab.j0 = dr_jmin;  slab.k0 = 0;
                slab.ni = dr_imax - dr_imin + 1; slab.nj = dr_jmax - dr_jmin + 1;
                                          slab.nk = LAP; break;
        case 5: slab.i0 = dr_imin;        slab.j0 = dr_jmin;  slab.k0 = NK - LAP;
                slab.ni = dr_imax - dr_imin + 1; slab.nj = dr_jmax - dr_jmin + 1;
                                          slab.nk = LAP; break;
        default: return slab;
        }

        PetscInt total = slab.ni * slab.nj * slab.nk;
        slab.x.resize((size_t)total);
        slab.y.resize((size_t)total);
        slab.z.resize((size_t)total);

        PetscInt idx = 0;
        for (PetscInt k = 0; k < slab.nk; ++k) {
            PetscInt kg = slab.k0 + k;
            for (PetscInt j = 0; j < slab.nj; ++j) {
                PetscInt jg = slab.j0 + j;
                for (PetscInt i = 0; i < slab.ni; ++i) {
                    PetscInt ig = slab.i0 + i;
                    PetscInt gi = (kg * NJ + jg) * NI + ig;
                    slab.x[idx] = coords.x[gi];
                    slab.y[idx] = coords.y[gi];
                    slab.z[idx] = coords.z[gi];
                    ++idx;
                }
            }
        }
        slab.valid = true;
        return slab;
    };

    // ======== 世界广播（只处理 is_periodic 连接）========
    PetscInt nConn = (PetscInt)face_connections_.size();
    for (PetscInt ci = 0; ci < nConn; ++ci) {
        auto& conn = face_connections_[ci];
        if (!conn.is_periodic) continue;

        // ── 方向1: block_a ← block_b ──
        {
            PetscMPIInt root = (PetscMPIInt)block_start_rank_[conn.block_b];
            PetscMPIInt dims[6];
            PetscMPIInt total = 0;
            std::vector<PetscReal> xb, yb, zb;

            if (rank == root) {
                DonorSlab s = packSlab((PetscInt)conn.block_b,
                                       (PetscInt)conn.face_b,
                                       conn.donorrange);
                if (s.valid) {
                    dims[0] = (PetscMPIInt)s.ni; dims[1] = (PetscMPIInt)s.nj;
                    dims[2] = (PetscMPIInt)s.nk; dims[3] = (PetscMPIInt)s.i0;
                    dims[4] = (PetscMPIInt)s.j0; dims[5] = (PetscMPIInt)s.k0;
                    total   = (PetscMPIInt)(s.ni * s.nj * s.nk);
                    xb = std::move(s.x);
                    yb = std::move(s.y);
                    zb = std::move(s.z);
                }
            }
            MPI_Bcast(dims,  6, MPI_INT,    root, PETSC_COMM_WORLD);
            MPI_Bcast(&total, 1, MPI_INT,    root, PETSC_COMM_WORLD);
            if (total > 0) {
                if (rank != root) { xb.resize((size_t)total); yb.resize((size_t)total); zb.resize((size_t)total); }
                MPI_Bcast(xb.data(), total, MPI_DOUBLE, root, PETSC_COMM_WORLD);
                MPI_Bcast(yb.data(), total, MPI_DOUBLE, root, PETSC_COMM_WORLD);
                MPI_Bcast(zb.data(), total, MPI_DOUBLE, root, PETSC_COMM_WORLD);
            }

            PetscMPIInt recv_root = (PetscMPIInt)block_start_rank_[conn.block_a];
            if (rank == recv_root) {
                DonorSlab s;
                s.ni = (PetscInt)dims[0]; s.nj = (PetscInt)dims[1];
                s.nk = (PetscInt)dims[2]; s.i0 = (PetscInt)dims[3];
                s.j0 = (PetscInt)dims[4]; s.k0 = (PetscInt)dims[5];
                s.valid = true;
                if (total > 0) { s.x = std::move(xb); s.y = std::move(yb); s.z = std::move(zb); }
                donor_slabs_[conn.block_a][conn.face_a] = std::move(s);
            }
        }

        // ── 方向2: block_b ← block_a ──
        {
            PetscMPIInt root = (PetscMPIInt)block_start_rank_[conn.block_a];
            PetscMPIInt dims[6];
            PetscMPIInt total = 0;
            std::vector<PetscReal> xb, yb, zb;

            if (rank == root) {
                DonorSlab s = packSlab((PetscInt)conn.block_a,
                                       (PetscInt)conn.face_a,
                                       conn.pointrange);
                if (s.valid) {
                    dims[0] = (PetscMPIInt)s.ni; dims[1] = (PetscMPIInt)s.nj;
                    dims[2] = (PetscMPIInt)s.nk; dims[3] = (PetscMPIInt)s.i0;
                    dims[4] = (PetscMPIInt)s.j0; dims[5] = (PetscMPIInt)s.k0;
                    total   = (PetscMPIInt)(s.ni * s.nj * s.nk);
                    xb = std::move(s.x);
                    yb = std::move(s.y);
                    zb = std::move(s.z);
                }
            }
            MPI_Bcast(dims,  6, MPI_INT,    root, PETSC_COMM_WORLD);
            MPI_Bcast(&total, 1, MPI_INT,    root, PETSC_COMM_WORLD);
            if (total > 0) {
                if (rank != root) { xb.resize((size_t)total); yb.resize((size_t)total); zb.resize((size_t)total); }
                MPI_Bcast(xb.data(), total, MPI_DOUBLE, root, PETSC_COMM_WORLD);
                MPI_Bcast(yb.data(), total, MPI_DOUBLE, root, PETSC_COMM_WORLD);
                MPI_Bcast(zb.data(), total, MPI_DOUBLE, root, PETSC_COMM_WORLD);
            }

            PetscMPIInt recv_root = (PetscMPIInt)block_start_rank_[conn.block_b];
            if (rank == recv_root) {
                DonorSlab s;
                s.ni = (PetscInt)dims[0]; s.nj = (PetscInt)dims[1];
                s.nk = (PetscInt)dims[2]; s.i0 = (PetscInt)dims[3];
                s.j0 = (PetscInt)dims[4]; s.k0 = (PetscInt)dims[5];
                s.valid = true;
                if (total > 0) { s.x = std::move(xb); s.y = std::move(yb); s.z = std::move(zb); }
                donor_slabs_[conn.block_b][conn.face_b] = std::move(s);
            }
        }
    }

    // ======== 各块内广播 ========
    for (PetscInt b = 0; b < num_blocks; ++b) {
        if ((PetscInt)my_block != b) continue;
        MPI_Comm comm = block_comms_[b];
        for (PetscMPIInt face = 0; face < 6; ++face) {
            DonorSlab& slab = donor_slabs_[b][face];
            PetscMPIInt vf = slab.valid ? 1 : 0;
            MPI_Bcast(&vf, 1, MPI_INT, 0, comm);
            slab.valid = (vf != 0);
            if (!slab.valid) continue;

            PetscMPIInt dims[6];
            PetscMPIInt total;
            if (rank == (PetscMPIInt)block_start_rank_[b]) {
                dims[0] = (PetscMPIInt)slab.ni; dims[1] = (PetscMPIInt)slab.nj;
                dims[2] = (PetscMPIInt)slab.nk; dims[3] = (PetscMPIInt)slab.i0;
                dims[4] = (PetscMPIInt)slab.j0; dims[5] = (PetscMPIInt)slab.k0;
                total   = (PetscMPIInt)(slab.ni * slab.nj * slab.nk);
            }
            MPI_Bcast(dims, 6, MPI_INT, 0, comm);
            MPI_Bcast(&total, 1, MPI_INT, 0, comm);

            if (rank != (PetscMPIInt)block_start_rank_[b]) {
                slab.ni = dims[0]; slab.nj = dims[1]; slab.nk = dims[2];
                slab.i0 = dims[3]; slab.j0 = dims[4]; slab.k0 = dims[5];
                if (total > 0) { slab.x.resize((size_t)total); slab.y.resize((size_t)total); slab.z.resize((size_t)total); }
            }
            if (total > 0) {
                MPI_Bcast(slab.x.data(), total, MPI_DOUBLE, 0, comm);
                MPI_Bcast(slab.y.data(), total, MPI_DOUBLE, 0, comm);
                MPI_Bcast(slab.z.data(), total, MPI_DOUBLE, 0, comm);
            }
        }
    }

    if (rank == 0)
        PetscPrintf(PETSC_COMM_WORLD, "[exchangeDonorSlabsPeriodic] done.\n");
}

// ====================================================================
// exchangeDonorSlabsUPeriodic — 为周期性连接打包守恒量 DonorSlab
// ====================================================================
void MultiBlockMesh::exchangeDonorSlabsUPeriodic()
{
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    PetscInt nConn = (PetscInt)face_connections_.size();
    if (nConn == 0) return;

    for (PetscInt ci = 0; ci < nConn; ++ci) {
        auto& conn = face_connections_[ci];
        if (!conn.is_periodic) continue;

        // ── 方向1: block_a ← block_b ──
        {
            PetscMPIInt root = (PetscMPIInt)block_start_rank_[conn.block_b];
            PetscMPIInt dims[6];
            PetscMPIInt total = 0;
            std::vector<PetscReal> ub[5];

            if (rank == root) {
                const auto& coords = full_coords_[conn.block_b];
                if (coords.flow_ready) {
                    const DonorSlab& ref = donor_slabs_[conn.block_b][conn.face_b];
                    if (ref.valid) {
                        dims[0] = (PetscMPIInt)ref.ni; dims[1] = (PetscMPIInt)ref.nj;
                        dims[2] = (PetscMPIInt)ref.nk; dims[3] = (PetscMPIInt)ref.i0;
                        dims[4] = (PetscMPIInt)ref.j0; dims[5] = (PetscMPIInt)ref.k0;
                        total = (PetscMPIInt)(ref.ni * ref.nj * ref.nk);
                        PetscInt NI = coords.ni, NJ = coords.nj;
                        for (int c = 0; c < 5; ++c) {
                            ub[c].resize((size_t)total);
                            PetscInt slot = 0;
                            for (PetscInt k = 0; k < ref.nk; ++k) {
                                PetscInt kg = ref.k0 + k;
                                for (PetscInt j = 0; j < ref.nj; ++j) {
                                    PetscInt jg = ref.j0 + j;
                                    for (PetscInt i = 0; i < ref.ni; ++i) {
                                        PetscInt ig = ref.i0 + i;
                                        PetscInt gi = ((size_t)kg * NJ + jg) * NI + ig;
                                        ub[c][slot++] = coords.u[c][gi];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            MPI_Bcast(dims,  6, MPI_INT,    root, PETSC_COMM_WORLD);
            MPI_Bcast(&total, 1, MPI_INT,    root, PETSC_COMM_WORLD);
            if (total > 0) {
                for (int c = 0; c < 5; ++c) {
                    if (rank != root) ub[c].resize((size_t)total);
                    MPI_Bcast(ub[c].data(), total, MPI_DOUBLE, root, PETSC_COMM_WORLD);
                }
            }

            PetscMPIInt recv_root = (PetscMPIInt)block_start_rank_[conn.block_a];
            if (rank == recv_root && total > 0) {
                auto& slab = donor_slabs_[conn.block_a][conn.face_a];
                for (int c = 0; c < 5; ++c)
                    slab.u[c] = std::move(ub[c]);
            }
        }

        // ── 方向2: block_b ← block_a ──
        {
            PetscMPIInt root = (PetscMPIInt)block_start_rank_[conn.block_a];
            PetscMPIInt dims[6];
            PetscMPIInt total = 0;
            std::vector<PetscReal> ub[5];

            if (rank == root) {
                const auto& coords = full_coords_[conn.block_a];
                if (coords.flow_ready) {
                    const DonorSlab& ref = donor_slabs_[conn.block_a][conn.face_a];
                    if (ref.valid) {
                        dims[0] = (PetscMPIInt)ref.ni; dims[1] = (PetscMPIInt)ref.nj;
                        dims[2] = (PetscMPIInt)ref.nk; dims[3] = (PetscMPIInt)ref.i0;
                        dims[4] = (PetscMPIInt)ref.j0; dims[5] = (PetscMPIInt)ref.k0;
                        total = (PetscMPIInt)(ref.ni * ref.nj * ref.nk);
                        PetscInt NI = coords.ni, NJ = coords.nj;
                        for (int c = 0; c < 5; ++c) {
                            ub[c].resize((size_t)total);
                            PetscInt slot = 0;
                            for (PetscInt k = 0; k < ref.nk; ++k) {
                                PetscInt kg = ref.k0 + k;
                                for (PetscInt j = 0; j < ref.nj; ++j) {
                                    PetscInt jg = ref.j0 + j;
                                    for (PetscInt i = 0; i < ref.ni; ++i) {
                                        PetscInt ig = ref.i0 + i;
                                        PetscInt gi = ((size_t)kg * NJ + jg) * NI + ig;
                                        ub[c][slot++] = coords.u[c][gi];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            MPI_Bcast(dims,  6, MPI_INT,    root, PETSC_COMM_WORLD);
            MPI_Bcast(&total, 1, MPI_INT,    root, PETSC_COMM_WORLD);
            if (total > 0) {
                for (int c = 0; c < 5; ++c) {
                    if (rank != root) ub[c].resize((size_t)total);
                    MPI_Bcast(ub[c].data(), total, MPI_DOUBLE, root, PETSC_COMM_WORLD);
                }
            }

            PetscMPIInt recv_root = (PetscMPIInt)block_start_rank_[conn.block_b];
            if (rank == recv_root && total > 0) {
                auto& slab = donor_slabs_[conn.block_b][conn.face_b];
                for (int c = 0; c < 5; ++c)
                    slab.u[c] = std::move(ub[c]);
            }
        }
    }

    // ======== 各块内广播 ========
    PetscInt num_blocks = getNumBlocks();
    PetscInt my_block = -1;
    for (PetscInt b = 0; b < num_blocks; ++b) {
        if (full_coords_[b].valid) { my_block = b; break; }
    }

    for (PetscInt b = 0; b < num_blocks; ++b) {
        if (my_block != b) continue;
        MPI_Comm comm = block_comms_[b];
        for (PetscMPIInt face = 0; face < 6; ++face) {
            DonorSlab& slab = donor_slabs_[b][face];
            if (!slab.valid) continue;

            PetscMPIInt total = (PetscMPIInt)(slab.ni * slab.nj * slab.nk);
            if (total <= 0) continue;
            if (rank != (PetscMPIInt)block_start_rank_[b])
                for (int c = 0; c < 5; ++c)
                    slab.u[c].resize((size_t)total);
            for (int c = 0; c < 5; ++c)
                MPI_Bcast(slab.u[c].data(), total, MPI_DOUBLE, 0, comm);
        }
    }

    if (rank == 0)
        PetscPrintf(PETSC_COMM_WORLD, "[exchangeDonorSlabsUPeriodic] done.\n");
}

// ====================================================================
// gatherFullMetrics — 从 DMDA 汇集各块完整度量系数到 full_coords_
// ====================================================================
void MultiBlockMesh::gatherFullMetrics()
{
    PetscMPIInt world_rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &world_rank);
    PetscInt num_blocks = getNumBlocks();

    for (PetscInt b = 0; b < num_blocks; ++b) {
        Mesh* blk = getBlock(b);
        if (!blk) continue;

        MPI_Comm comm = block_comms_[b];
        int rank_in_block, size_in_block;
        MPI_Comm_rank(comm, &rank_in_block);
        MPI_Comm_size(comm, &size_in_block);

        PetscInt nxg = blk->getNxGlobal();
        PetscInt nyg = blk->getNyGlobal();
        PetscInt nzg = blk->getNzGlobal();
        PetscInt ntotal = nxg * nyg * nzg;

        DM da = blk->getDM();
        DMDALocalInfo info;
        DMDAGetLocalInfo(da, &info);

        PetscInt xs = info.xs, ys = info.ys, zs = info.zs;
        PetscInt xm = info.xm, ym = info.ym, zm = info.zm;
        PetscInt nlocal = xm * ym * zm;

        auto metricVecs = blk->getGlobalMetricVecs();
        for (int comp = 0; comp < 10; ++comp)
            full_coords_[b].m[comp].resize((size_t)ntotal);

        for (int comp = 0; comp < 10; ++comp) {
            PetscReal*** arr;
            DMDAVecGetArray(da, metricVecs[comp], &arr);

            std::vector<PetscReal> local_buf((size_t)nlocal);
            PetscInt idx = 0;
            for (PetscInt k = zs; k < zs + zm; ++k)
                for (PetscInt j = ys; j < ys + ym; ++j)
                    for (PetscInt i = xs; i < xs + xm; ++i)
                        local_buf[idx++] = arr[k][j][i];

            DMDAVecRestoreArray(da, metricVecs[comp], &arr);

            PetscInt my_meta[6] = {xs, ys, zs, xm, ym, zm};
            std::vector<PetscInt> all_meta;
            if (rank_in_block == 0)
                all_meta.resize((size_t)size_in_block * 6);
            MPI_Gather(my_meta, 6, MPIU_INT,
                       all_meta.data(), 6, MPIU_INT, 0, comm);

            int my_n = (int)nlocal;
            std::vector<int> all_n;
            if (rank_in_block == 0)
                all_n.resize((size_t)size_in_block);
            MPI_Gather(&my_n, 1, MPI_INT,
                       all_n.data(), 1, MPI_INT, 0, comm);

            std::vector<int> displs;
            std::vector<PetscReal> all_data;
            if (rank_in_block == 0) {
                int total = 0;
                displs.resize((size_t)size_in_block);
                for (int r = 0; r < size_in_block; ++r) {
                    displs[r] = total;
                    total += all_n[r];
                }
                all_data.resize((size_t)total);
            }
            MPI_Gatherv(local_buf.data(), my_n, MPI_DOUBLE,
                        all_data.data(), all_n.data(), displs.data(),
                        MPI_DOUBLE, 0, comm);

            if (rank_in_block == 0) {
                auto& fm = full_coords_[b].m[comp];
                for (int r = 0; r < size_in_block; ++r) {
                    PetscInt* m = &all_meta[(size_t)r * 6];
                    PetscInt r_xs = m[0], r_ys = m[1], r_zs = m[2];
                    PetscInt r_xm = m[3], r_ym = m[4], r_zm = m[5];
                    PetscReal* src = all_data.data() + displs[r];
                    PetscInt src_idx = 0;
                    for (PetscInt k = r_zs; k < r_zs + r_zm; ++k)
                        for (PetscInt j = r_ys; j < r_ys + r_ym; ++j)
                            for (PetscInt i = r_xs; i < r_xs + r_xm; ++i)
                                fm[(size_t)(k * nyg + j) * nxg + i] = src[src_idx++];
                }
            }

            MPI_Bcast(full_coords_[b].m[comp].data(), (int)ntotal,
                      MPI_DOUBLE, 0, comm);
        }

        full_coords_[b].metrics_ready = true;
    }

    if (world_rank == 0)
        PetscPrintf(PETSC_COMM_WORLD, "[gatherFullMetrics] done.\n");
}

// ====================================================================
// exchangeDonorSlabsMetricsPeriodic — 为周期性连接打包度量系数 DonorSlab
// ====================================================================
void MultiBlockMesh::exchangeDonorSlabsMetricsPeriodic()
{
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    PetscInt nConn = (PetscInt)face_connections_.size();
    if (nConn == 0) return;

    // 确定本进程属于哪个块
    PetscInt num_blocks = getNumBlocks();
    PetscInt my_block = -1;
    for (PetscInt b = 0; b < num_blocks; ++b) {
        if (full_coords_[b].valid) { my_block = b; break; }
    }

    for (PetscInt ci = 0; ci < nConn; ++ci) {
        auto& conn = face_connections_[ci];
        if (!conn.is_periodic) continue;

        // ── 方向1: block_a ← block_b ──
        {
            PetscMPIInt root = (PetscMPIInt)block_start_rank_[conn.block_b];
            PetscMPIInt dims[6] = {0};
            PetscMPIInt total = 0;
            std::vector<PetscReal> mb[10];

            if (rank == root) {
                const auto& coords = full_coords_[conn.block_b];
                if (coords.metrics_ready) {
                    const DonorSlab& ref = donor_slabs_[conn.block_b][conn.face_b];
                    if (ref.valid) {
                        dims[0] = (PetscMPIInt)ref.ni; dims[1] = (PetscMPIInt)ref.nj;
                        dims[2] = (PetscMPIInt)ref.nk; dims[3] = (PetscMPIInt)ref.i0;
                        dims[4] = (PetscMPIInt)ref.j0; dims[5] = (PetscMPIInt)ref.k0;
                        total = (PetscMPIInt)(ref.ni * ref.nj * ref.nk);
                        PetscInt NI = coords.ni, NJ = coords.nj;
                        for (int c = 0; c < 10; ++c) {
                            mb[c].resize((size_t)total);
                            PetscInt slot = 0;
                            for (PetscInt k = 0; k < ref.nk; ++k) {
                                PetscInt kg = ref.k0 + k;
                                for (PetscInt j = 0; j < ref.nj; ++j) {
                                    PetscInt jg = ref.j0 + j;
                                    for (PetscInt i = 0; i < ref.ni; ++i) {
                                        PetscInt ig = ref.i0 + i;
                                        PetscInt gi = ((size_t)kg * NJ + jg) * NI + ig;
                                        mb[c][slot++] = coords.m[c][gi];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            MPI_Bcast(dims,  6, MPI_INT,    root, PETSC_COMM_WORLD);
            MPI_Bcast(&total, 1, MPI_INT,    root, PETSC_COMM_WORLD);
            if (total > 0) {
                for (int c = 0; c < 10; ++c) {
                    if (rank != root) mb[c].resize((size_t)total);
                    MPI_Bcast(mb[c].data(), total, MPI_DOUBLE, root, PETSC_COMM_WORLD);
                }
            }
            // ★ 所有属于 block_a 的进程都存，不只 recv_root
            if (total > 0 && my_block == conn.block_a) {
                auto& slab = donor_slabs_[conn.block_a][conn.face_a];
                for (int c = 0; c < 10; ++c)
                    slab.m[c] = mb[c];
            }
        }

        // ── 方向2: block_b ← block_a ──
        {
            PetscMPIInt root = (PetscMPIInt)block_start_rank_[conn.block_a];
            PetscMPIInt dims[6] = {0};
            PetscMPIInt total = 0;
            std::vector<PetscReal> mb[10];

            if (rank == root) {
                const auto& coords = full_coords_[conn.block_a];
                if (coords.metrics_ready) {
                    const DonorSlab& ref = donor_slabs_[conn.block_a][conn.face_a];
                    if (ref.valid) {
                        dims[0] = (PetscMPIInt)ref.ni; dims[1] = (PetscMPIInt)ref.nj;
                        dims[2] = (PetscMPIInt)ref.nk; dims[3] = (PetscMPIInt)ref.i0;
                        dims[4] = (PetscMPIInt)ref.j0; dims[5] = (PetscMPIInt)ref.k0;
                        total = (PetscMPIInt)(ref.ni * ref.nj * ref.nk);
                        PetscInt NI = coords.ni, NJ = coords.nj;
                        for (int c = 0; c < 10; ++c) {
                            mb[c].resize((size_t)total);
                            PetscInt slot = 0;
                            for (PetscInt k = 0; k < ref.nk; ++k) {
                                PetscInt kg = ref.k0 + k;
                                for (PetscInt j = 0; j < ref.nj; ++j) {
                                    PetscInt jg = ref.j0 + j;
                                    for (PetscInt i = 0; i < ref.ni; ++i) {
                                        PetscInt ig = ref.i0 + i;
                                        PetscInt gi = ((size_t)kg * NJ + jg) * NI + ig;
                                        mb[c][slot++] = coords.m[c][gi];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            MPI_Bcast(dims,  6, MPI_INT,    root, PETSC_COMM_WORLD);
            MPI_Bcast(&total, 1, MPI_INT,    root, PETSC_COMM_WORLD);
            if (total > 0) {
                for (int c = 0; c < 10; ++c) {
                    if (rank != root) mb[c].resize((size_t)total);
                    MPI_Bcast(mb[c].data(), total, MPI_DOUBLE, root, PETSC_COMM_WORLD);
                }
            }
            // ★ 所有属于 block_b 的进程都存
            if (total > 0 && my_block == conn.block_b) {
                auto& slab = donor_slabs_[conn.block_b][conn.face_b];
                for (int c = 0; c < 10; ++c)
                    slab.m[c] = mb[c];
            }
        }
    }

    if (rank == 0)
        PetscPrintf(PETSC_COMM_WORLD, "[exchangeDonorSlabsMetricsPeriodic] done.\n");
}

// ====================================================================
// applyBCOverrides — 用户定义的 BC 覆写 CGNS 默认值
// ====================================================================
void MultiBlockMesh::applyBCOverrides(const std::vector<BCOverride>& overrides)
{
    if (overrides.empty()) return;

    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    for (const auto& ov : overrides) {
        // ★ 检查 CGNS 1-to-1 冲突：不能对内部连接面定义物理 BC
        {
            const char* fn[6] = {"LEFT","RIGHT","BOTTOM","TOP","BACK","FRONT"};
            for (const auto& fc : face_connections_) {
                if (fc.is_periodic) continue;
                if ((fc.block_a == ov.block && fc.face_a == ov.face) ||
                    (fc.block_b == ov.block && fc.face_b == ov.face)) {
                    throw std::runtime_error(
                        std::string("BC override error: block ") + std::to_string(ov.block) +
                        " face " + fn[ov.face] + " is a CGNS 1-to-1 internal connection, "
                        "cannot assign physical BC '" + ov.bc_type + "'");
                }
            }
        }

        bool found = false;
        for (auto& fbc : face_bcs_) {
            if (fbc.block == ov.block && fbc.face == ov.face) {
                if (rank == 0)
                    PetscPrintf(PETSC_COMM_WORLD,
                        "[applyBCOverrides] block %d face %d: %s → %s\n",
                        ov.block, ov.face, fbc.bc_type.c_str(), ov.bc_type.c_str());
                fbc.bc_type = ov.bc_type;
                found = true;
                break;
            }
        }
        if (!found) {
            // 新增（CGNS 里没有这条 BC 记录）
            FaceBC fbc;
            fbc.block   = ov.block;
            fbc.face    = ov.face;
            fbc.bc_type = ov.bc_type;
            face_bcs_.push_back(fbc);
            if (rank == 0)
                PetscPrintf(PETSC_COMM_WORLD,
                    "[applyBCOverrides] block %d face %d: (none) → %s (new)\n",
                    ov.block, ov.face, ov.bc_type.c_str());
        }
    }

    if (rank == 0)
        PetscPrintf(PETSC_COMM_WORLD, "[applyBCOverrides] %d overrides applied.\n",
                    (int)overrides.size());
}

// ====================================================================
// applyPeriodic — 注册周期性边界对（含校验）
// ====================================================================
void MultiBlockMesh::applyPeriodic(const std::vector<PeriodicPair>& pairs)
{
    if (pairs.empty()) return;

    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    const char* fn[6] = {"LEFT","RIGHT","BOTTOM","TOP","BACK","FRONT"};
    int num_blocks = (int)blocks_.size();

    for (const auto& pp : pairs) {
        // ── 1. block 索引存在性 ──
        if (pp.block_a < 0 || pp.block_a >= num_blocks ||
            pp.block_b < 0 || pp.block_b >= num_blocks)
            throw std::runtime_error(
                std::string("Periodic error: block ") + std::to_string(pp.block_a) +
                " or " + std::to_string(pp.block_b) + " out of range [0," +
                std::to_string(num_blocks-1) + "]");

        // ── 2. 成对检查 ──
        bool has_reverse = false;
        for (const auto& pp2 : pairs) {
            if (pp2.block_a == pp.block_b && pp2.face_a == pp.face_b &&
                pp2.block_b == pp.block_a && pp2.face_b == pp.face_a) {
                has_reverse = true;
                break;
            }
        }
        if (!has_reverse)
            throw std::runtime_error(
                std::string("Periodic error: block ") + std::to_string(pp.block_a) +
                " " + fn[pp.face_a] + " -> block " + std::to_string(pp.block_b) +
                " " + fn[pp.face_b] + " has no reverse pair");

        // ── 3. CGNS 1-to-1 冲突 ──
        for (const auto& fc : face_connections_) {
            if (fc.is_periodic) continue;
            if ((fc.block_a == pp.block_a && fc.face_a == pp.face_a) ||
                (fc.block_b == pp.block_a && fc.face_b == pp.face_a) ||
                (fc.block_a == pp.block_b && fc.face_a == pp.face_b) ||
                (fc.block_b == pp.block_b && fc.face_b == pp.face_b))
                throw std::runtime_error(
                    std::string("Periodic error: block ") + std::to_string(pp.block_a) +
                    " " + fn[pp.face_a] + " or block " + std::to_string(pp.block_b) +
                    " " + fn[pp.face_b] + " is a CGNS 1-to-1 internal connection");
        }

        // ── 4. 切向尺寸检查 ──
        Mesh* blk_a = getBlock(pp.block_a);
        Mesh* blk_b = getBlock(pp.block_b);
        if (blk_a && blk_b) {
            PetscInt nxa = blk_a->getNxGlobal(), nya = blk_a->getNyGlobal(), nza = blk_a->getNzGlobal();
            PetscInt nxb = blk_b->getNxGlobal(), nyb = blk_b->getNyGlobal(), nzb = blk_b->getNzGlobal();

            int fn_a = pp.face_a / 2;  // 0=i, 1=j, 2=k
            int fn_b = pp.face_b / 2;

            if (fn_a != fn_b)
                throw std::runtime_error(
                    std::string("Periodic error: block ") + std::to_string(pp.block_a) +
                    " " + fn[pp.face_a] + " and block " + std::to_string(pp.block_b) +
                    " " + fn[pp.face_b] + " have different normal directions");

            PetscInt ta1, ta2, tb1, tb2;
            if (fn_a == 0)      { ta1 = nya; ta2 = nza; tb1 = nyb; tb2 = nzb; }
            else if (fn_a == 1) { ta1 = nxa; ta2 = nza; tb1 = nxb; tb2 = nzb; }
            else                { ta1 = nxa; ta2 = nya; tb1 = nxb; tb2 = nyb; }

            if (ta1 != tb1 || ta2 != tb2)
                throw std::runtime_error(
                    std::string("Periodic error: tangential dimension mismatch — block ") +
                    std::to_string(pp.block_a) + " " + fn[pp.face_a] + " (" +
                    std::to_string(ta1) + "×" + std::to_string(ta2) + ") vs block " +
                    std::to_string(pp.block_b) + " " + fn[pp.face_b] + " (" +
                    std::to_string(tb1) + "×" + std::to_string(tb2) + ")");
        }

        // ── 5. 构造全脸 FaceConnectivity ──
        FaceConnectivity fc;
        fc.block_a = pp.block_a;  fc.face_a = pp.face_a;
        fc.block_b = pp.block_b;  fc.face_b = pp.face_b;
        fc.is_periodic = true;
        fc.transform[0] = 1;
        fc.transform[1] = 2;
        fc.transform[2] = 3;
        fc.translate[0] = pp.translate[0];
        fc.translate[1] = pp.translate[1];
        fc.translate[2] = pp.translate[2];

        auto fillRange = [](cgsize_t* pr, int face, PetscInt ni, PetscInt nj, PetscInt nk) {
            switch (face) {
            case 0: pr[0]=0; pr[1]=0;     pr[2]=0; pr[3]=nj-1; pr[4]=0; pr[5]=nk-1; break;
            case 1: pr[0]=ni-1; pr[1]=ni-1; pr[2]=0; pr[3]=nj-1; pr[4]=0; pr[5]=nk-1; break;
            case 2: pr[0]=0; pr[1]=ni-1; pr[2]=0; pr[3]=0;     pr[4]=0; pr[5]=nk-1; break;
            case 3: pr[0]=0; pr[1]=ni-1; pr[2]=nj-1; pr[3]=nj-1; pr[4]=0; pr[5]=nk-1; break;
            case 4: pr[0]=0; pr[1]=ni-1; pr[2]=0; pr[3]=nj-1; pr[4]=0; pr[5]=0;     break;
            case 5: pr[0]=0; pr[1]=ni-1; pr[2]=0; pr[3]=nj-1; pr[4]=nk-1; pr[5]=nk-1; break;
            }
        };

        PetscInt nia=0,nja=0,nka=0, nib=0,njb=0,nkb=0;
        if (blk_a) { nia = blk_a->getNxGlobal(); nja = blk_a->getNyGlobal(); nka = blk_a->getNzGlobal(); }
        if (blk_b) { nib = blk_b->getNxGlobal(); njb = blk_b->getNyGlobal(); nkb = blk_b->getNzGlobal(); }

        fillRange(fc.pointrange, pp.face_a, nia, nja, nka);
        fillRange(fc.donorrange,  pp.face_b, nib, njb, nkb);

        face_connections_.push_back(fc);

        if (rank == 0)
            PetscPrintf(PETSC_COMM_WORLD,
                "[applyPeriodic] block %d %s ↔ block %d %s\n",
                pp.block_a, fn[pp.face_a], pp.block_b, fn[pp.face_b]);
    }

    if (rank == 0)
        PetscPrintf(PETSC_COMM_WORLD, "[applyPeriodic] %d pairs registered.\n",
                    (int)pairs.size());
}
