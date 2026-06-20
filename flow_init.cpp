#include "flow_init.h"
#include "mesh.h"
#include "mesh_mutiblock.h"
#include "sim_config.h"
#include <petsc.h>
#include <cmath>
#include <cstdio>
#include <cstdint>

// ====================================================================
// farfieldInit — 无穷远均匀来流
// ====================================================================
static PetscErrorCode farfieldInit(MultiBlockMesh& multiMesh, const SimConfig& cfg)
{
    PetscReal Ma    = cfg.getMach();
    PetscReal gamma = cfg.getGamma();
    PetscReal AoA   = cfg.getAttack()  * M_PI / 180.0;
    PetscReal beta  = cfg.getSideslip() * M_PI / 180.0;

    PetscReal d_val = 1.0;
    PetscReal u_val = cos(AoA) * cos(beta);
    PetscReal v_val = sin(AoA) * cos(beta);
    PetscReal w_val = sin(beta);
    PetscReal p_val = 1.0 / (gamma * Ma * Ma);

    PetscReal U[5];
    U[0] = d_val;
    U[1] = d_val * u_val;
    U[2] = d_val * v_val;
    U[3] = d_val * w_val;
    U[4] = p_val / (gamma - 1.0)
         + 0.5 * d_val * (u_val*u_val + v_val*v_val + w_val*w_val);

    PetscInt numBlocks = multiMesh.getNumBlocks();
    PetscErrorCode ierr;

    for (PetscInt b = 0; b < numBlocks; ++b) {
        Mesh* blk = multiMesh.getBlock(b);
        if (!blk) continue;

        DM da_c = blk->getDM();
        DMDALocalInfo info;
        ierr = DMDAGetLocalInfo(da_c, &info); CHKERRQ(ierr);

        PetscInt xs = info.xs, ys = info.ys, zs = info.zs;
        PetscInt xm = info.xm, ym = info.ym, zm = info.zm;

        for (int comp = 0; comp < 5; ++comp) {
            PetscReal*** arr = nullptr;
            ierr = DMDAVecGetArray(da_c, blk->getURef(comp), &arr); CHKERRQ(ierr);
            for (PetscInt k = zs; k < zs + zm; ++k)
                for (PetscInt j = ys; j < ys + ym; ++j)
                    for (PetscInt i = xs; i < xs + xm; ++i)
                        arr[k][j][i] = U[comp];
            ierr = DMDAVecRestoreArray(da_c, blk->getURef(comp), &arr); CHKERRQ(ierr);
        }
        ierr = blk->syncUGlobalToLocal(); CHKERRQ(ierr);
    }

    PetscPrintf(PETSC_COMM_WORLD,
        "[farfieldInit] d=%.6f u=%.6f v=%.6f w=%.6f p=%.6f "
        "U1=%.6f U2=%.6f U3=%.6f U4=%.6f U5=%.6f\n",
        d_val, u_val, v_val, w_val, p_val,
        U[0], U[1], U[2], U[3], U[4]);

    return 0;
}

// ====================================================================
// sinusoidalInit
// M-gM-^TM-(M-hM-/M-^]M-eM-^GM- M-dM-8M--M-gM-^NM-^PM-eM-^OM-^BM-gM-^ZM-^DM-fM--M-#M-eM-<M-&M-fM-3M-\M-eM-^JM-(M-fM-^@M-^AM-eM-^LM-^VM-eM-^LM- M-eM-^IM-^MM-eM-^PM-^QM-eM-^LM-^VM-dM-8M-^JM-fM-^IM- 10% M-fM--M-#M-eM-<M-&M-fM-^SM-8M-eM-^JM-(
// ====================================================================
static PetscErrorCode sinusoidalInit(MultiBlockMesh& multiMesh, const SimConfig& cfg)
{
    PetscReal Ma    = cfg.getMach();
    PetscReal gamma = cfg.getGamma();
    PetscReal AoA   = cfg.getAttack()  * M_PI / 180.0;
    PetscReal beta  = cfg.getSideslip() * M_PI / 180.0;

    PetscReal rho_inf = 1.0;
    PetscReal u_inf   = cos(AoA) * cos(beta);
    PetscReal v_inf   = sin(AoA) * cos(beta);
    PetscReal w_inf   = sin(beta);
    PetscReal p_inf   = 1.0 / (gamma * Ma * Ma);

    PetscReal kx = cfg.getKx();
    PetscReal ky = cfg.getKy();
    PetscReal kz = cfg.getKz();
    // k=0 M-hM-!M-(M-gM-^PM- M-fM-^\M-^IM-eM-^EM-(M-fM-^UM-0M-eM-^SM- M-dM-8M-^@M-dM-8M-*M-fM-3M-\M-eM-^QM-(M-eM-^EM-(M-fM-^PM-^QM-fM-^UM-0
    if (kx == 0.0) kx = 1.0;
    if (ky == 0.0) ky = 1.0;
    if (kz == 0.0) kz = 1.0;

    PetscInt numBlocks = multiMesh.getNumBlocks();
    PetscErrorCode ierr;
    PetscReal twopi = 2.0 * M_PI;

    // ---- M-hM-5M-<M-gM-;M-^YM-fM- M-<M-eM-?M-^CM-eM-^GM- M-dM-8M-- ----
    for (PetscInt b = 0; b < numBlocks; ++b) {
        Mesh* blk = multiMesh.getBlock(b);
        if (!blk) continue;

        PetscInt NX = blk->getNxGlobal();
        PetscInt NY = blk->getNyGlobal();
        PetscInt NZ = blk->getNzGlobal();

        DM da_c = blk->getDM();
        DMDALocalInfo info;
        ierr = DMDAGetLocalInfo(da_c, &info); CHKERRQ(ierr);

        PetscInt xs = info.xs, ys = info.ys, zs = info.zs;
        PetscInt xm = info.xm, ym = info.ym, zm = info.zm;

        PetscReal*** u0, ***u1, ***u2, ***u3, ***u4;
        ierr = DMDAVecGetArray(da_c, blk->getURef(0), &u0); CHKERRQ(ierr);
        ierr = DMDAVecGetArray(da_c, blk->getURef(1), &u1); CHKERRQ(ierr);
        ierr = DMDAVecGetArray(da_c, blk->getURef(2), &u2); CHKERRQ(ierr);
        ierr = DMDAVecGetArray(da_c, blk->getURef(3), &u3); CHKERRQ(ierr);
        ierr = DMDAVecGetArray(da_c, blk->getURef(4), &u4); CHKERRQ(ierr);

        PetscReal invNX = (NX > 1) ? 1.0 / (PetscReal)(NX - 1) : 0.0;
        PetscReal invNY = (NY > 1) ? 1.0 / (PetscReal)(NY - 1) : 0.0;
        PetscReal invNZ = (NZ > 1) ? 1.0 / (PetscReal)(NZ - 1) : 0.0;

        for (PetscInt k = zs; k < zs + zm; ++k) {
            PetscReal zk = (PetscReal)k * invNZ;
            for (PetscInt j = ys; j < ys + ym; ++j) {
                PetscReal yj = (PetscReal)j * invNY;
                for (PetscInt i = xs; i < xs + xm; ++i) {
                    PetscReal xi = (PetscReal)i * invNX;

                    PetscReal sx = sin(twopi * kx * xi);
                    PetscReal sy = sin(twopi * ky * yj);
                    PetscReal sz = sin(twopi * kz * zk);
                    PetscReal cx = cos(twopi * kx * xi);
                    PetscReal cy = cos(twopi * ky * yj);
                    PetscReal cz = cos(twopi * kz * zk);

                    PetscReal rho = rho_inf * (1.0 + 0.1 * sx * sy * sz);
                    PetscReal u   = u_inf   + 0.1 * sin(twopi * kx * xi + 1.0) * cy * cz;
                    PetscReal v   = v_inf   + 0.1 * cx * sin(twopi * ky * yj + 1.0) * cz;
                    PetscReal w   = w_inf   + 0.05* cx * cy * sin(twopi * kz * zk);
                    PetscReal p   = p_inf * (1.0 + 0.1 * sin(twopi * kx * xi + 0.5)
                                                   * sin(twopi * ky * yj + 0.5)
                                                   * sin(twopi * kz * zk + 0.5));

                    PetscReal gm1 = gamma - 1.0;
                    u0[k][j][i] = rho;
                    u1[k][j][i] = rho * u;
                    u2[k][j][i] = rho * v;
                    u3[k][j][i] = rho * w;
                    u4[k][j][i] = p / gm1 + 0.5 * rho * (u*u + v*v + w*w);
                }
            }
        }
        ierr = DMDAVecRestoreArray(da_c, blk->getURef(0), &u0); CHKERRQ(ierr);
        ierr = DMDAVecRestoreArray(da_c, blk->getURef(1), &u1); CHKERRQ(ierr);
        ierr = DMDAVecRestoreArray(da_c, blk->getURef(2), &u2); CHKERRQ(ierr);
        ierr = DMDAVecRestoreArray(da_c, blk->getURef(3), &u3); CHKERRQ(ierr);
        ierr = DMDAVecRestoreArray(da_c, blk->getURef(4), &u4); CHKERRQ(ierr);

        ierr = blk->syncUGlobalToLocal(); CHKERRQ(ierr);
    }

    PetscPrintf(PETSC_COMM_WORLD,
        "[sinusoidalInit] k_per_dir: kx=%.2f ky=%.2f kz=%.2f  (waves across domain)\n"
        "  d_inf=%.4f u_inf=%.4f v_inf=%.4f w_inf=%.4f p_inf=%.4f\n",
        (double)kx, (double)ky, (double)kz,
        (double)rho_inf, (double)u_inf, (double)v_inf, (double)w_inf, (double)p_inf);

    return 0;
}

// ====================================================================
// datInit — 从二进制 dat 文件读取格心物理场
// 格式：
//   int32_t N                                 — 块数
//   对每个块：
//     int32_t ni, nj, nk                     — 格心维度
//     double  rho[ni*nj*nk]                  — 5 个分量
//     double  rhou[ni*nj*nk]
//     double  rhov[ni*nj*nk]
//     double  rhow[ni*nj*nk]
//     double  rhoE[ni*nj*nk]
// ====================================================================
static PetscErrorCode datInit(MultiBlockMesh& multiMesh, const std::string& filename)
{
    PetscErrorCode ierr;
    FILE* fp = fopen(filename.c_str(), "rb");
    if (!fp) {
        PetscPrintf(PETSC_COMM_WORLD, "[datInit] ERROR: cannot open %s\n", filename.c_str());
        return 1;
    }

    int32_t N;
    if (fread(&N, sizeof(int32_t), 1, fp) != 1) {
        fclose(fp);
        return 1;
    }

    PetscInt numBlocks = multiMesh.getNumBlocks();
    if ((PetscInt)N != numBlocks) {
        PetscPrintf(PETSC_COMM_WORLD,
            "[datInit] ERROR: file has %d blocks, mesh has %d\n", (int)N, (int)numBlocks);
        fclose(fp);
        return 1;
    }

    std::vector<double> buf;

    for (int32_t b = 0; b < N; ++b) {
        int32_t ni, nj, nk;
        if (fread(&ni, sizeof(int32_t), 1, fp) != 1 ||
            fread(&nj, sizeof(int32_t), 1, fp) != 1 ||
            fread(&nk, sizeof(int32_t), 1, fp) != 1) {
            fclose(fp); return 1;
        }

        Mesh* blk = multiMesh.getBlock(b);
        if (!blk) {
            // 不拥有该块 → 跳过 5 个分量的数据，保持多进程文件指针同步
            size_t skip = 5 * (size_t)ni * (size_t)nj * (size_t)nk * sizeof(double);
            fseek(fp, (long)skip, SEEK_CUR);
            continue;
        }

        if ((PetscInt)ni != blk->getNxGlobal() ||
            (PetscInt)nj != blk->getNyGlobal() ||
            (PetscInt)nk != blk->getNzGlobal()) {
            PetscPrintf(PETSC_COMM_WORLD,
                "[datInit] ERROR: block %d size mismatch: file (%d,%d,%d) vs mesh (%d,%d,%d)\n",
                (int)b, (int)ni, (int)nj, (int)nk,
                (int)blk->getNxGlobal(), (int)blk->getNyGlobal(), (int)blk->getNzGlobal());
            fclose(fp); return 1;
        }

        size_t total = (size_t)ni * (size_t)nj * (size_t)nk;
        buf.resize(total);

        DM da_c = blk->getDM();

        for (int comp = 0; comp < 5; ++comp) {
            if (fread(buf.data(), sizeof(double), total, fp) != total) {
                fclose(fp); return 1;
            }

            DMDALocalInfo info;
            ierr = DMDAGetLocalInfo(da_c, &info); CHKERRQ(ierr);

            PetscReal*** arr = nullptr;
            ierr = DMDAVecGetArray(da_c, blk->getURef(comp), &arr); CHKERRQ(ierr);

            PetscInt xs = info.xs, ys = info.ys, zs = info.zs;
            PetscInt xm = info.xm, ym = info.ym, zm = info.zm;

            for (PetscInt k = zs; k < zs + zm; ++k)
                for (PetscInt j = ys; j < ys + ym; ++j)
                    for (PetscInt i = xs; i < xs + xm; ++i) {
                        size_t idx = ((size_t)k * (size_t)nj + (size_t)j) * (size_t)ni + (size_t)i;
                        arr[k][j][i] = (PetscReal)buf[idx];
                    }

            ierr = DMDAVecRestoreArray(da_c, blk->getURef(comp), &arr); CHKERRQ(ierr);
        }

        ierr = blk->syncUGlobalToLocal(); CHKERRQ(ierr);
    }

    fclose(fp);
    PetscPrintf(PETSC_COMM_WORLD, "[datInit] loaded %d blocks from %s\n", (int)N, filename.c_str());
    return 0;
}

// ====================================================================
// initializeFlowField — 统一入口
// ====================================================================
PetscErrorCode initializeFlowField(MultiBlockMesh& mesh, const SimConfig& cfg)
{
    std::string itype = cfg.getInitType();
    if (itype == "farfield")
        return farfieldInit(mesh, cfg);
    else if (itype == "sinusoidal")
        return sinusoidalInit(mesh, cfg);
    else if (itype == "dat")
        return datInit(mesh, cfg.getInitFile());
    else {
        PetscPrintf(PETSC_COMM_WORLD,
            "[init] ERROR: unknown init_type '%s' (expect farfield or dat)\n", itype.c_str());
        return 1;
    }
}

// ====================================================================
// writeRestart — 写出二进制重启文件（格心守恒量），格式与 datInit 匹配
// 格式：
//   int32_t N                                 — 块数
//   对每个块：
//     int32_t ni, nj, nk                     — 格心维度
//     double  rho[ni*nj*nk]                  — 5 个分量
//     double  rhou[ni*nj*nk]
//     double  rhov[ni*nj*nk]
//     double  rhow[ni*nj*nk]
//     double  rhoE[ni*nj*nk]
// ====================================================================
PetscErrorCode writeRestart(MultiBlockMesh& mesh, const std::string& filename)
{
    PetscMPIInt world_rank, world_size;
    MPI_Comm_rank(PETSC_COMM_WORLD, &world_rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &world_size);
    PetscInt nb = mesh.getNumBlocks();

    // World rank 0 打开文件、写块数
    FILE* fp = nullptr;
    if (world_rank == 0) {
        fp = fopen(filename.c_str(), "wb");
        if (!fp) {
            PetscPrintf(PETSC_COMM_WORLD,
                "[writeRestart] ERROR: cannot open %s\n", filename.c_str());
            return 1;
        }
        int32_t N = (int32_t)nb;
        fwrite(&N, sizeof(int32_t), 1, fp);
    }

    for (PetscInt b = 0; b < nb; ++b) {
        // 找到该块的 root rank（同 ExportAllCellFlowToTecplot 模式）
        PetscMPIInt have_block = (mesh.getBlockComm(b) != MPI_COMM_NULL) ? 1 : 0;
        std::vector<PetscMPIInt> all_have(world_size);
        MPI_Allgather(&have_block, 1, MPI_INT, all_have.data(), 1, MPI_INT, PETSC_COMM_WORLD);
        PetscMPIInt root_rank = -1;
        for (int r = 0; r < world_size; ++r) {
            if (all_have[r]) { root_rank = r; break; }
        }
        if (root_rank < 0) continue;

        // 打包缓冲区：header(3 double) + u[0..4]
        std::vector<double> sendbuf;
        int32_t ni = 0, nj = 0, nk = 0;

        if (world_rank == root_rank) {
            const auto& fc = mesh.getFullCoords(b);
            if (!fc.flow_ready) {
                PetscPrintf(PETSC_COMM_WORLD,
                    "[writeRestart] ERROR: block %d flow not gathered\n", (int)b);
                if (world_rank == 0) fclose(fp);
                return 1;
            }
            ni = fc.ni; nj = fc.nj; nk = fc.nk;
            size_t total = (size_t)ni * nj * nk;
            sendbuf.resize(3 + 5 * total);
            sendbuf[0] = (double)ni;
            sendbuf[1] = (double)nj;
            sendbuf[2] = (double)nk;
            for (int comp = 0; comp < 5; ++comp) {
                size_t off = 3 + (size_t)comp * total;
                for (size_t i = 0; i < total; ++i)
                    sendbuf[off + i] = (double)fc.u[comp][i];
            }
        }

        // root_rank → world rank 0（若不同）
        if (root_rank != 0) {
            if (world_rank == root_rank) {
                int sz = (int)sendbuf.size();
                MPI_Send(&sz, 1, MPI_INT, 0, (int)(b * 2), PETSC_COMM_WORLD);
                MPI_Send(sendbuf.data(), sz, MPI_DOUBLE, 0, (int)(b * 2 + 1), PETSC_COMM_WORLD);
            } else if (world_rank == 0) {
                int sz;
                MPI_Status st;
                MPI_Recv(&sz, 1, MPI_INT, root_rank, (int)(b * 2), PETSC_COMM_WORLD, &st);
                sendbuf.resize((size_t)sz);
                MPI_Recv(sendbuf.data(), sz, MPI_DOUBLE, root_rank, (int)(b * 2 + 1), PETSC_COMM_WORLD, &st);
                ni = (int32_t)sendbuf[0];
                nj = (int32_t)sendbuf[1];
                nk = (int32_t)sendbuf[2];
            }
        }

        // World rank 0 写入本块数据
        if (world_rank == 0) {
            fwrite(&ni, sizeof(int32_t), 1, fp);
            fwrite(&nj, sizeof(int32_t), 1, fp);
            fwrite(&nk, sizeof(int32_t), 1, fp);
            size_t total = (size_t)ni * nj * nk;
            for (int comp = 0; comp < 5; ++comp)
                fwrite(sendbuf.data() + 3 + comp * total, sizeof(double), total, fp);
        }
    }

    if (world_rank == 0) {
        fclose(fp);
        PetscPrintf(PETSC_COMM_WORLD,
            "[writeRestart] wrote %d blocks to %s\n", (int)nb, filename.c_str());
    }
    return 0;
}
