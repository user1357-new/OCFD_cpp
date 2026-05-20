#include "mesh.h"
#include "mesh_mutiblock.h"
#include <petscdmda.h>
#include <petscvec.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <mpi.h>
#include <vector>
#include <string>

// 构造函数
Mesh::Mesh(PetscInt nx_g, PetscInt ny_g, PetscInt nz_g,
           PetscInt my_id_, PetscInt npx0_, PetscInt npy0_, PetscInt npz0_,
           PetscInt lap, PetscInt grid_type, PetscInt scheme_vis, MPI_Comm comm_)
    : comm(comm_),
      da(nullptr),
      nx_global(nx_g), ny_global(ny_g), nz_global(nz_g),
      nx(0), ny(0), nz(0),
      my_id(my_id_),
      npx0(npx0_), npy0(npy0_), npz0(npz0_),
      LAP(lap),
      Iflag_Gridtype(grid_type),
      Scheme_Vis(scheme_vis),
      Axx(nullptr), Ayy(nullptr), Azz(nullptr),
      Axx_local(nullptr), Ayy_local(nullptr), Azz_local(nullptr),
      Akx(nullptr), Aky(nullptr), Akz(nullptr),
      Aix(nullptr), Aiy(nullptr), Aiz(nullptr),
      Asx(nullptr), Asy(nullptr), Asz(nullptr),
      Ajac(nullptr),
      Akx1(nullptr), Aky1(nullptr), Akz1(nullptr),
      Aix1(nullptr), Aiy1(nullptr), Aiz1(nullptr),
      Asx1(nullptr), Asy1(nullptr), Asz1(nullptr),
      Akx_local(nullptr), Aky_local(nullptr), Akz_local(nullptr),
      Aix_local(nullptr), Aiy_local(nullptr), Aiz_local(nullptr),
      Asx_local(nullptr), Asy_local(nullptr), Asz_local(nullptr),
      Ajac_local(nullptr),  
      local_pool_initialized(false),
      boundary_condition_handler(nullptr)
{
}

Mesh::~Mesh()
{
    // 销毁持久化 local vector 池（13个）
    if (Axx_local) { VecDestroy(&Axx_local); Axx_local = nullptr; }
    if (Ayy_local) { VecDestroy(&Ayy_local); Ayy_local = nullptr; }
    if (Azz_local) { VecDestroy(&Azz_local); Azz_local = nullptr; }
    
    if (Akx_local) { VecDestroy(&Akx_local); Akx_local = nullptr; }
    if (Aky_local) { VecDestroy(&Aky_local); Aky_local = nullptr; }
    if (Akz_local) { VecDestroy(&Akz_local); Akz_local = nullptr; }
    if (Aix_local) { VecDestroy(&Aix_local); Aix_local = nullptr; }
    if (Aiy_local) { VecDestroy(&Aiy_local); Aiy_local = nullptr; }
    if (Aiz_local) { VecDestroy(&Aiz_local); Aiz_local = nullptr; }
    if (Asx_local) { VecDestroy(&Asx_local); Asx_local = nullptr; }
    if (Asy_local) { VecDestroy(&Asy_local); Asy_local = nullptr; }
    if (Asz_local) { VecDestroy(&Asz_local); Asz_local = nullptr; }
    if (Ajac_local) { VecDestroy(&Ajac_local); Ajac_local = nullptr; }
    
    // 销毁全局向量
    if (Axx) VecDestroy(&Axx);
    if (Ayy) VecDestroy(&Ayy);
    if (Azz) VecDestroy(&Azz);
    
    if (Akx) VecDestroy(&Akx);
    if (Aky) VecDestroy(&Aky);
    if (Akz) VecDestroy(&Akz);
    if (Aix) VecDestroy(&Aix);
    if (Aiy) VecDestroy(&Aiy);
    if (Aiz) VecDestroy(&Aiz);
    if (Asx) VecDestroy(&Asx);
    if (Asy) VecDestroy(&Asy);
    if (Asz) VecDestroy(&Asz);
    if (Ajac) VecDestroy(&Ajac);
    
    if (Akx1) VecDestroy(&Akx1);
    if (Aky1) VecDestroy(&Aky1);
    if (Akz1) VecDestroy(&Akz1);
    if (Aix1) VecDestroy(&Aix1);
    if (Aiy1) VecDestroy(&Aiy1);
    if (Aiz1) VecDestroy(&Aiz1);
    if (Asx1) VecDestroy(&Asx1);
    if (Asy1) VecDestroy(&Asy1);
    if (Asz1) VecDestroy(&Asz1);
    
    // 销毁DM
    if (da) DMDestroy(&da);
}

// 获取局部数组指针
PetscErrorCode Mesh::GetLocalArrays(
    PetscReal*** &Axx_arr, PetscReal*** &Ayy_arr, PetscReal*** &Azz_arr,
    PetscReal*** &Akx_arr, PetscReal*** &Aky_arr, PetscReal*** &Akz_arr,
    PetscReal*** &Aix_arr, PetscReal*** &Aiy_arr, PetscReal*** &Aiz_arr,
    PetscReal*** &Asx_arr, PetscReal*** &Asy_arr, PetscReal*** &Asz_arr,
    PetscReal*** &Ajac_arr,
    PetscReal*** &Akx1_arr, PetscReal*** &Aky1_arr, PetscReal*** &Akz1_arr,
    PetscReal*** &Aix1_arr, PetscReal*** &Aiy1_arr, PetscReal*** &Aiz1_arr,
    PetscReal*** &Asx1_arr, PetscReal*** &Asy1_arr, PetscReal*** &Asz1_arr,
    DMDALocalInfo &info
)
{
    PetscErrorCode ierr;
    
    ierr = DMDAGetLocalInfo(da, &info);
    CHKERRQ(ierr);
    
    ierr = DMDAVecGetArray(da, Axx, &Axx_arr);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Ayy, &Ayy_arr);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Azz, &Azz_arr);
    CHKERRQ(ierr);
    
    ierr = DMDAVecGetArray(da, Akx, &Akx_arr);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Aky, &Aky_arr);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Akz, &Akz_arr);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Aix, &Aix_arr);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Aiy, &Aiy_arr);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Aiz, &Aiz_arr);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Asx, &Asx_arr);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Asy, &Asy_arr);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Asz, &Asz_arr);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Ajac, &Ajac_arr);
    CHKERRQ(ierr);
    
    ierr = DMDAVecGetArray(da, Akx1, &Akx1_arr);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Aky1, &Aky1_arr);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Akz1, &Akz1_arr);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Aix1, &Aix1_arr);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Aiy1, &Aiy1_arr);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Aiz1, &Aiz1_arr);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Asx1, &Asx1_arr);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Asy1, &Asy1_arr);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Asz1, &Asz1_arr);
    CHKERRQ(ierr);
    
    return 0;
}

// 释放局部数组指针
PetscErrorCode Mesh::RestoreLocalArrays(
    PetscReal*** &Axx_arr, PetscReal*** &Ayy_arr, PetscReal*** &Azz_arr,
    PetscReal*** &Akx_arr, PetscReal*** &Aky_arr, PetscReal*** &Akz_arr,
    PetscReal*** &Aix_arr, PetscReal*** &Aiy_arr, PetscReal*** &Aiz_arr,
    PetscReal*** &Asx_arr, PetscReal*** &Asy_arr, PetscReal*** &Asz_arr,
    PetscReal*** &Ajac_arr,
    PetscReal*** &Akx1_arr, PetscReal*** &Aky1_arr, PetscReal*** &Akz1_arr,
    PetscReal*** &Aix1_arr, PetscReal*** &Aiy1_arr, PetscReal*** &Aiz1_arr,
    PetscReal*** &Asx1_arr, PetscReal*** &Asy1_arr, PetscReal*** &Asz1_arr
)
{
    PetscErrorCode ierr;
    
    ierr = DMDAVecRestoreArray(da, Axx, &Axx_arr);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Ayy, &Ayy_arr);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Azz, &Azz_arr);
    CHKERRQ(ierr);
    
    ierr = DMDAVecRestoreArray(da, Akx, &Akx_arr);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Aky, &Aky_arr);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Akz, &Akz_arr);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Aix, &Aix_arr);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Aiy, &Aiy_arr);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Aiz, &Aiz_arr);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Asx, &Asx_arr);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Asy, &Asy_arr);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Asz, &Asz_arr);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Ajac, &Ajac_arr);
    CHKERRQ(ierr);
    
    ierr = DMDAVecRestoreArray(da, Akx1, &Akx1_arr);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Aky1, &Aky1_arr);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Akz1, &Akz1_arr);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Aix1, &Aix1_arr);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Aiy1, &Aiy1_arr);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Aiz1, &Aiz1_arr);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Asx1, &Asx1_arr);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Asy1, &Asy1_arr);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Asz1, &Asz1_arr);
    CHKERRQ(ierr);
    
    return 0;
}

// 初始化网格
PetscErrorCode Mesh::InitializeFromCoordinates(
    const std::vector<PetscReal> &x_global,
    const std::vector<PetscReal> &y_global,
    const std::vector<PetscReal> &z_global)
{
    PetscErrorCode ierr;

    // 1. 创建 DMDA（使用子通信域 comm）
    ierr = DMDACreate3d(comm,
                        DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED,
                        DMDA_STENCIL_BOX,
                        nx_global, ny_global, nz_global,
                        npx0, npy0, npz0,
                        1, LAP,
                        NULL, NULL, NULL,
                        &da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);

    // 2. 创建坐标向量
    ierr = DMCreateGlobalVector(da, &Axx); CHKERRQ(ierr);
    ierr = VecDuplicate(Axx, &Ayy); CHKERRQ(ierr);
    ierr = VecDuplicate(Axx, &Azz); CHKERRQ(ierr);

    DMDALocalInfo info;
    ierr = DMDAGetLocalInfo(da, &info); CHKERRQ(ierr);
    nx = info.xm; ny = info.ym; nz = info.zm;

    PetscReal ***axx, ***ayy, ***azz;
    ierr = DMDAVecGetArray(da, Axx, &axx); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Ayy, &ayy); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Azz, &azz); CHKERRQ(ierr);

    // 3. 填充局部坐标（全局索引：info.xs/ys/zs + 偏移）
    for (PetscInt k = info.zs; k < info.zs + info.zm; k++) {
        for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
            for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
                PetscInt idx = (k * ny_global + j) * nx_global + i;
                axx[k][j][i] = x_global[idx];
                ayy[k][j][i] = y_global[idx];
                azz[k][j][i] = z_global[idx];
            }
        }
    }

    ierr = DMDAVecRestoreArray(da, Axx, &axx); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Ayy, &ayy); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Azz, &azz); CHKERRQ(ierr);
    
    // 计算雅可比（同时创建持久化 local vector）
    ierr = comput_Jacobian3d(); CHKERRQ(ierr);

    // 统一创建并同步持久化 local vector 池（坐标 + 度量系数）
    ierr = ensureLocalVectors(); CHKERRQ(ierr);
    ierr = syncGlobalToLocal(); CHKERRQ(ierr);
    PetscPrintf(comm, "Initial Mesh OK (from coordinates)\n");
    return 0;
}

// ====================================================================
// ensureLocalVectors - 创建所有持久化 local vector（仅首次）
// 坐标：3个（Axx_local, Ayy_local, Azz_local）
// 度量系数：10个（Akx_local ~ Ajac_local）
// ====================================================================
PetscErrorCode Mesh::ensureLocalVectors()
{
    PetscErrorCode ierr;
    if (local_pool_initialized) return 0;

    // 创建坐标 local vectors
    ierr = DMCreateLocalVector(da, &Axx_local); CHKERRQ(ierr);
    ierr = VecDuplicate(Axx_local, &Ayy_local); CHKERRQ(ierr);
    ierr = VecDuplicate(Axx_local, &Azz_local); CHKERRQ(ierr);

    // 创建度量系数 local vectors
    ierr = VecDuplicate(Axx_local, &Akx_local); CHKERRQ(ierr);
    ierr = VecDuplicate(Axx_local, &Aky_local); CHKERRQ(ierr);
    ierr = VecDuplicate(Axx_local, &Akz_local); CHKERRQ(ierr);
    ierr = VecDuplicate(Axx_local, &Aix_local); CHKERRQ(ierr);
    ierr = VecDuplicate(Axx_local, &Aiy_local); CHKERRQ(ierr);
    ierr = VecDuplicate(Axx_local, &Aiz_local); CHKERRQ(ierr);
    ierr = VecDuplicate(Axx_local, &Asx_local); CHKERRQ(ierr);
    ierr = VecDuplicate(Axx_local, &Asy_local); CHKERRQ(ierr);
    ierr = VecDuplicate(Axx_local, &Asz_local); CHKERRQ(ierr);
    ierr = VecDuplicate(Axx_local, &Ajac_local); CHKERRQ(ierr);

    local_pool_initialized = true;
    return 0;
}

// ====================================================================
// syncGlobalToLocal - 将所有 global 向量同步到持久化 local vector
// 只更新物理域部分，ghost 层值保持不变
// ====================================================================
PetscErrorCode Mesh::syncGlobalToLocal()
{
    PetscErrorCode ierr;
    if (!local_pool_initialized) {
        ierr = ensureLocalVectors(); CHKERRQ(ierr);
    }

    // 坐标
    ierr = DMGlobalToLocalBegin(da, Axx, INSERT_VALUES, Axx_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, Axx, INSERT_VALUES, Axx_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da, Ayy, INSERT_VALUES, Ayy_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, Ayy, INSERT_VALUES, Ayy_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da, Azz, INSERT_VALUES, Azz_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, Azz, INSERT_VALUES, Azz_local); CHKERRQ(ierr);

    // 度量系数
    ierr = DMGlobalToLocalBegin(da, Akx, INSERT_VALUES, Akx_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, Akx, INSERT_VALUES, Akx_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da, Aky, INSERT_VALUES, Aky_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, Aky, INSERT_VALUES, Aky_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da, Akz, INSERT_VALUES, Akz_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, Akz, INSERT_VALUES, Akz_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da, Aix, INSERT_VALUES, Aix_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, Aix, INSERT_VALUES, Aix_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da, Aiy, INSERT_VALUES, Aiy_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, Aiy, INSERT_VALUES, Aiy_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da, Aiz, INSERT_VALUES, Aiz_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, Aiz, INSERT_VALUES, Aiz_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da, Asx, INSERT_VALUES, Asx_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, Asx, INSERT_VALUES, Asx_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da, Asy, INSERT_VALUES, Asy_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, Asy, INSERT_VALUES, Asy_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da, Asz, INSERT_VALUES, Asz_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, Asz, INSERT_VALUES, Asz_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da, Ajac, INSERT_VALUES, Ajac_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, Ajac, INSERT_VALUES, Ajac_local); CHKERRQ(ierr);
    return 0;
}

// ====================================================================
// getLocalCoordinateArrays - 获取持久化坐标 local vector 的数组指针（含 ghost）
// ====================================================================
PetscErrorCode Mesh::getLocalCoordinateArrays(PetscReal*** &axx, PetscReal*** &ayy, PetscReal*** &azz,
                                              DMDALocalInfo &info)
{
    PetscErrorCode ierr;
    if (!local_pool_initialized) {
        ierr = ensureLocalVectors(); CHKERRQ(ierr);
    }

    ierr = DMDAGetLocalInfo(da, &info); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Axx_local, &axx); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Ayy_local, &ayy); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Azz_local, &azz); CHKERRQ(ierr);
    return 0;
}

// ====================================================================
// restoreLocalCoordinateArrays - 恢复持久化坐标 local vector 的数组指针
// ====================================================================
PetscErrorCode Mesh::restoreLocalCoordinateArrays(PetscReal*** &axx, PetscReal*** &ayy, PetscReal*** &azz)
{
    PetscErrorCode ierr;
    ierr = DMDAVecRestoreArray(da, Axx_local, &axx); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Ayy_local, &ayy); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Azz_local, &azz); CHKERRQ(ierr);
    return 0;
}

// ====================================================================
// getLocalMetricArrays - 获取持久化度量系数 local vector 的数组指针
// ====================================================================
PetscErrorCode Mesh::getLocalMetricArrays(PetscReal*** &akx, PetscReal*** &aky, PetscReal*** &akz,
                                          PetscReal*** &aix, PetscReal*** &aiy, PetscReal*** &aiz,
                                          PetscReal*** &asx, PetscReal*** &asy, PetscReal*** &asz,
                                          PetscReal*** &ajac, DMDALocalInfo &info)
{
    PetscErrorCode ierr;
    if (!local_pool_initialized) {
        ierr = ensureLocalVectors(); CHKERRQ(ierr);
    }

    ierr = DMDAGetLocalInfo(da, &info); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Akx_local, &akx); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Aky_local, &aky); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Akz_local, &akz); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Aix_local, &aix); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Aiy_local, &aiy); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Aiz_local, &aiz); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Asx_local, &asx); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Asy_local, &asy); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Asz_local, &asz); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Ajac_local, &ajac); CHKERRQ(ierr);
    return 0;
}

// ====================================================================
// restoreLocalMetricArrays - 恢复持久化度量系数 local vector 的数组指针
// ====================================================================
PetscErrorCode Mesh::restoreLocalMetricArrays(PetscReal*** &akx, PetscReal*** &aky, PetscReal*** &akz,
                                              PetscReal*** &aix, PetscReal*** &aiy, PetscReal*** &aiz,
                                              PetscReal*** &asx, PetscReal*** &asy, PetscReal*** &asz,
                                              PetscReal*** &ajac)
{
    PetscErrorCode ierr;
    ierr = DMDAVecRestoreArray(da, Akx_local, &akx); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Aky_local, &aky); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Akz_local, &akz); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Aix_local, &aix); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Aiy_local, &aiy); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Aiz_local, &aiz); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Asx_local, &asx); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Asy_local, &asy); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Asz_local, &asz); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Ajac_local, &ajac); CHKERRQ(ierr);
    return 0;
}
// ====================================================================
// getLocalCoordinateVecs - 返回坐标持久化 local Vec（3个）
// ====================================================================
std::vector<Vec> Mesh::getLocalCoordinateVecs() const
{
    if (!local_pool_initialized) return {};
    return {Axx_local, Ayy_local, Azz_local};
}

// ====================================================================
// getLocalMetricVecs - 返回度量系数持久化 local Vec（10个）
// ====================================================================
std::vector<Vec> Mesh::getLocalMetricVecs() const
{
    if (!local_pool_initialized) return {};
    return {Akx_local, Aky_local, Akz_local,
            Aix_local, Aiy_local, Aiz_local,
            Asx_local, Asy_local, Asz_local,
            Ajac_local};
}
