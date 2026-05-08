#include "mesh.h"
#include <petscdmda.h>
#include <petscvec.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <mpi.h>
#include <vector>
#include <string> 

//构造函数
Mesh::Mesh(PetscInt nx_g, PetscInt ny_g, PetscInt nz_g,
           PetscInt my_id_, 
           PetscInt npx0_, PetscInt npy0_, PetscInt npz0_,
           PetscInt lap, PetscInt grid_type, PetscInt scheme_vis,
           MPI_Comm comm_)
    : nx_global(nx_g), ny_global(ny_g), nz_global(nz_g),
      my_id(my_id_), 
      npx0(npx0_), npy0(npy0_), npz0(npz0_),
      LAP(lap), Iflag_Gridtype(grid_type), Scheme_Vis(scheme_vis),
      comm(comm_),                  
      da(nullptr),
      Axx(nullptr), Ayy(nullptr), Azz(nullptr),
      Akx(nullptr), Aky(nullptr), Akz(nullptr),
      Aix(nullptr), Aiy(nullptr), Aiz(nullptr),
      Asx(nullptr), Asy(nullptr), Asz(nullptr),
      Ajac(nullptr),
      Akx1(nullptr), Aky1(nullptr), Akz1(nullptr),
      Aix1(nullptr), Aiy1(nullptr), Aiz1(nullptr),
      Asx1(nullptr), Asy1(nullptr), Asz1(nullptr),
      boundary_condition_handler(nullptr)
{
}

// 析构函数
Mesh::~Mesh()
{
    if (Asz1) VecDestroy(&Asz1);
    if (Asy1) VecDestroy(&Asy1);
    if (Asx1) VecDestroy(&Asx1);
    if (Aiz1) VecDestroy(&Aiz1);
    if (Aiy1) VecDestroy(&Aiy1);
    if (Aix1) VecDestroy(&Aix1);
    if (Akz1) VecDestroy(&Akz1);
    if (Aky1) VecDestroy(&Aky1);
    if (Akx1) VecDestroy(&Akx1);
    
    if (Ajac) VecDestroy(&Ajac);
    if (Asz) VecDestroy(&Asz);
    if (Asy) VecDestroy(&Asy);
    if (Asx) VecDestroy(&Asx);
    if (Aiz) VecDestroy(&Aiz);
    if (Aiy) VecDestroy(&Aiy);
    if (Aix) VecDestroy(&Aix);
    if (Akz) VecDestroy(&Akz);
    if (Aky) VecDestroy(&Aky);
    if (Akx) VecDestroy(&Akx);
    
    if (Azz) VecDestroy(&Azz);
    if (Ayy) VecDestroy(&Ayy);
    if (Axx) VecDestroy(&Axx);
    
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
                        DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
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
    //计算jacobi
    ierr = comput_Jacobian3d(); CHKERRQ(ierr);

    ierr = Jac_Ghost_boundary(); CHKERRQ(ierr);

    PetscPrintf(comm, "Initial Mesh OK (from coordinates)\n");
    return 0;
}