#include "mesh.h"
#include <petscdmda.h>
#include <petscvec.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <mpi.h>

// 构造函数
Mesh::Mesh(PetscInt nx_g, PetscInt ny_g, PetscInt nz_g,
           PetscInt my_id_, 
           PetscInt npx0_, PetscInt npy0_, PetscInt npz0_,
           PetscInt lap, PetscInt grid_type, PetscInt scheme_vis)
    : nx_global(nx_g), ny_global(ny_g), nz_global(nz_g),
      my_id(my_id_), 
      npx0(npx0_), npy0(npy0_), npz0(npz0_),
      LAP(lap), Iflag_Gridtype(grid_type), Scheme_Vis(scheme_vis),
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
    // 删除了 i_offset, j_offset, k_offset 的手动分配和计算
    // 因为我们将直接使用 PETSc 提供的 info.xs/ys/zs 作为全局索引
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

// 注册边界条件处理函数
void Mesh::RegisterBoundaryConditionHandler(std::function<void()> handler)
{
    boundary_condition_handler = handler;
}

// 公开初始化接口
PetscErrorCode Mesh::Initialize()
{
    return init_mesh();
}

// 初始化网格（对应原init_mesh，完全一致的逻辑）
PetscErrorCode Mesh::init_mesh()
{
    PetscErrorCode ierr;
    
    // Step 1: 根据网格类型读取网格
    if (Iflag_Gridtype == GRID1D) {
        ierr = read_mesh1d();
        CHKERRQ(ierr);
    } else if (Iflag_Gridtype == GRID2D_PLANE) {
        ierr = read_mesh2d_plane();
        CHKERRQ(ierr);
    } else if (Iflag_Gridtype == GRID2D_AXIAL_SYMM) {
        ierr = read_mesh2d_AxialSymm();
        CHKERRQ(ierr);
    } else if (Iflag_Gridtype == GRID3D || Iflag_Gridtype == GRID_AND_JACOBIAN3D) {
        ierr = read_mesh3d();
        CHKERRQ(ierr);
    }
    
    // Step 2: 交换边界数据（MPI部分，留空给您实现）
    exchange_boundary_xyz(Axx);
    exchange_boundary_xyz(Ayy);
    exchange_boundary_xyz(Azz);
    
    // Step 3: 调用用户注册的边界条件处理函数（包括周期性边界等）
    if (boundary_condition_handler) {
        boundary_condition_handler();
    }
    
    // Step 4: 如果不是预计算的雅可比，则计算
    if (Iflag_Gridtype != GRID_AND_JACOBIAN3D) {
        ierr = comput_Jacobian3d();
        CHKERRQ(ierr);
    }
    
    // Step 5: 交换雅可比系数边界（MPI部分，留空给您实现）
    exchange_boundary_xyz(Akx);
    exchange_boundary_xyz(Aky);
    exchange_boundary_xyz(Akz);
    exchange_boundary_xyz(Aix);
    exchange_boundary_xyz(Aiy);
    exchange_boundary_xyz(Aiz);
    exchange_boundary_xyz(Asx);
    exchange_boundary_xyz(Asy);
    exchange_boundary_xyz(Asz);
    exchange_boundary_xyz(Ajac);
    
    // Step 6: Ghost Cell边界处理（MPI部分，留空给您实现）
    ierr = Jac_Ghost_boundary();
    CHKERRQ(ierr);
    
    if (my_id == 0) {
        PetscPrintf(PETSC_COMM_WORLD, "Initial Mesh OK\n");
    }
    
    return 0;
}

// 读取1D网格（对应原read_mesh1d）
PetscErrorCode Mesh::read_mesh1d()
{
    PetscErrorCode ierr;
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    
    // 分配临时数组存储全局网格
    PetscReal *x0 = new PetscReal[nx_global];
    PetscReal *y0 = new PetscReal[ny_global];
    PetscReal *z0 = new PetscReal[nz_global];
    
    // 进程0读取文件
    if (rank == 0) {
        PetscPrintf(PETSC_COMM_WORLD, "read 1D mesh ...\n");
        
        std::ifstream file("OCFD-grid.dat", std::ios::binary | std::ios::in);
        if (!file.is_open()) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
                    "Cannot open OCFD-grid.dat");
        }
        
        file.read(reinterpret_cast<char*>(x0), nx_global * sizeof(PetscReal));
        file.read(reinterpret_cast<char*>(y0), ny_global * sizeof(PetscReal));
        file.read(reinterpret_cast<char*>(z0), nz_global * sizeof(PetscReal));
        
        file.close();
    }
    
    // 广播到所有进程
    MPI_Bcast(x0, nx_global, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    MPI_Bcast(y0, ny_global, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    MPI_Bcast(z0, nz_global, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    
    // 创建DMDA：使用您规定的 npx0, npy0, npz0 进行分块
    ierr = DMDACreate3d(PETSC_COMM_WORLD,
                        DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                        DMDA_STENCIL_BOX,
                        nx_global, ny_global, nz_global,
                        npx0, npy0, npz0,  // ← 这里使用了您规定的参数
                        1, LAP,
                        NULL, NULL, NULL,
                        &da);
    CHKERRQ(ierr);
    
    ierr = DMSetUp(da);
    CHKERRQ(ierr);
    
    // 创建坐标向量
    ierr = DMCreateGlobalVector(da, &Axx);
    CHKERRQ(ierr);
    ierr = VecDuplicate(Axx, &Ayy);
    CHKERRQ(ierr);
    ierr = VecDuplicate(Axx, &Azz);
    CHKERRQ(ierr);
    
    // 获取局部信息
    DMDALocalInfo info;
    ierr = DMDAGetLocalInfo(da, &info);
    CHKERRQ(ierr);
    
    nx = info.xm; ny = info.ym; nz = info.zm;
    
    PetscReal ***axx, ***ayy, ***azz;
    ierr = DMDAVecGetArray(da, Axx, &axx);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Ayy, &ayy);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Azz, &azz);
    CHKERRQ(ierr);
    
    // 简化逻辑：直接使用 info.xs/ys/zs 作为全局索引访问 x0/y0/z0
    for (PetscInt k = info.zs; k < info.zs + info.zm; k++) {
        for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
            for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
                axx[k][j][i] = x0[i];
                ayy[k][j][i] = y0[j];
                azz[k][j][i] = z0[k];
            }
        }
    }
    
    ierr = DMDAVecRestoreArray(da, Axx, &axx);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Ayy, &ayy);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Azz, &azz);
    CHKERRQ(ierr);
    
    delete[] x0;
    delete[] y0;
    delete[] z0;
    
    return 0;
}

// 读取2D平面网格
PetscErrorCode Mesh::read_mesh2d_plane()
{
    PetscErrorCode ierr;
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    
    PetscReal *x2 = new PetscReal[nx_global * ny_global];
    PetscReal *y2 = new PetscReal[nx_global * ny_global];
    PetscReal *z1 = new PetscReal[nz_global];
    
    if (rank == 0) {
        PetscPrintf(PETSC_COMM_WORLD, "read 2D plane mesh ...\n");
        
        std::ifstream file("OCFD-grid.dat", std::ios::binary | std::ios::in);
        if (!file.is_open()) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
                    "Cannot open OCFD-grid.dat");
        }
        
        // 跳过 Fortran unformatted 记录标记（每个记录前后有4字节长度）
        int record_len;
        
        // 读取 x2d
        file.read(reinterpret_cast<char*>(&record_len), 4);
        file.read(reinterpret_cast<char*>(x2), nx_global * ny_global * sizeof(PetscReal));
        file.read(reinterpret_cast<char*>(&record_len), 4);
        
        // 读取 y2d
        file.read(reinterpret_cast<char*>(&record_len), 4);
        file.read(reinterpret_cast<char*>(y2), nx_global * ny_global * sizeof(PetscReal));
        file.read(reinterpret_cast<char*>(&record_len), 4);
        
        // 读取 z1d
        file.read(reinterpret_cast<char*>(&record_len), 4);
        file.read(reinterpret_cast<char*>(z1), nz_global * sizeof(PetscReal));
        file.read(reinterpret_cast<char*>(&record_len), 4);
        
        file.close();
    }
    
    MPI_Bcast(x2, nx_global * ny_global, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    MPI_Bcast(y2, nx_global * ny_global, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    MPI_Bcast(z1, nz_global, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    
    ierr = DMDACreate3d(PETSC_COMM_WORLD,
                        DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                        DMDA_STENCIL_BOX,
                        nx_global, ny_global, nz_global,
                        npx0, npy0, npz0, // 使用规定分块
                        1, LAP,
                        NULL, NULL, NULL,
                        &da);
    CHKERRQ(ierr);
    
    ierr = DMSetUp(da);
    CHKERRQ(ierr);
    

    ierr = DMCreateGlobalVector(da, &Axx);
    CHKERRQ(ierr);
    ierr = VecDuplicate(Axx, &Ayy);
    CHKERRQ(ierr);
    ierr = VecDuplicate(Axx, &Azz);
    CHKERRQ(ierr);
    
    DMDALocalInfo info;
    ierr = DMDAGetLocalInfo(da, &info);
    CHKERRQ(ierr);
    
    nx = info.xm; ny = info.ym; nz = info.zm;
    
// ... existing code ...
    nx = info.xm; ny = info.ym; nz = info.zm;
    
    PetscPrintf(PETSC_COMM_WORLD, "Rank %d: xs=%d, xm=%d, ys=%d, ym=%d, zs=%d, zm=%d\n", 
                rank, info.xs, info.xm, info.ys, info.ym, info.zs, info.zm);
    



    PetscReal ***axx, ***ayy, ***azz;
    ierr = DMDAVecGetArray(da, Axx, &axx);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Ayy, &ayy);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Azz, &azz);
    CHKERRQ(ierr);
    
    for (PetscInt k = info.zs; k < info.zs + info.zm; k++) {
        for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
            for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
                
                PetscInt idx = j * nx_global + i;
                axx[k][j][i] = x2[idx];
                ayy[k][j][i] = y2[idx];
                azz[k][j][i] = z1[k];
             
            }
        }
    }
    
    ierr = DMDAVecRestoreArray(da, Axx, &axx);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Ayy, &ayy);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Azz, &azz);
    CHKERRQ(ierr);
    
    delete[] x2;
    delete[] y2;
    delete[] z1;
    
    return 0;
}

// 读取2D轴对称网格
PetscErrorCode Mesh::read_mesh2d_AxialSymm()
{
    PetscErrorCode ierr;
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    
    PetscReal *x2 = new PetscReal[nx_global * ny_global];
    PetscReal *R2 = new PetscReal[nx_global * ny_global];
    PetscReal *seta1 = new PetscReal[nz_global];
    
    if (rank == 0) {
        PetscPrintf(PETSC_COMM_WORLD, "read 2D axial symm mesh ...\n");
        
        std::ifstream file("OCFD-grid.dat", std::ios::binary | std::ios::in);
        if (!file.is_open()) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
                    "Cannot open OCFD-grid.dat");
        }
        
        file.read(reinterpret_cast<char*>(x2), 
                  nx_global * ny_global * sizeof(PetscReal));
        file.read(reinterpret_cast<char*>(R2), 
                  nx_global * ny_global * sizeof(PetscReal));
        file.read(reinterpret_cast<char*>(seta1), 
                  nz_global * sizeof(PetscReal));
        
        file.close();
    }
    
    MPI_Bcast(x2, nx_global * ny_global, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    MPI_Bcast(R2, nx_global * ny_global, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    MPI_Bcast(seta1, nz_global, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    
    ierr = DMDACreate3d(PETSC_COMM_WORLD,
                        DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                        DMDA_STENCIL_BOX,
                        nx_global, ny_global, nz_global,
                        npx0, npy0, npz0, // 使用规定分块
                        1, LAP,
                        NULL, NULL, NULL,
                        &da);
    CHKERRQ(ierr);
    
    ierr = DMSetUp(da);
    CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(da, &Axx);
    CHKERRQ(ierr);
    ierr = VecDuplicate(Axx, &Ayy);
    CHKERRQ(ierr);
    ierr = VecDuplicate(Axx, &Azz);
    CHKERRQ(ierr);
    
    DMDALocalInfo info;
    ierr = DMDAGetLocalInfo(da, &info);
    CHKERRQ(ierr);
    
    nx = info.xm; ny = info.ym; nz = info.zm;
    
    PetscReal ***axx, ***ayy, ***azz;
    ierr = DMDAVecGetArray(da, Axx, &axx);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Ayy, &ayy);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Azz, &azz);
    CHKERRQ(ierr);
    
    for (PetscInt k = info.zs; k < info.zs + info.zm; k++) {
        for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
            for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
                PetscInt idx = j * nx_global + i;
                axx[k][j][i] = x2[idx];
                ayy[k][j][i] = R2[idx] * std::cos(seta1[k]);
                azz[k][j][i] = R2[idx] * std::sin(seta1[k]);
            }
        }
    }
    
    ierr = DMDAVecRestoreArray(da, Axx, &axx);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Ayy, &ayy);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Azz, &azz);
    CHKERRQ(ierr);
    
    delete[] x2;
    delete[] R2;
    delete[] seta1;
    
    return 0;
}

// 读取3D网格
PetscErrorCode Mesh::read_mesh3d()
{
    PetscErrorCode ierr;
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    
    // 分配临时数组存储全局网格
    PetscReal *x3 = new PetscReal[nx_global * ny_global * nz_global];
    PetscReal *y3 = new PetscReal[nx_global * ny_global * nz_global];
    PetscReal *z3 = new PetscReal[nx_global * ny_global * nz_global];
    
    if (rank == 0) {
        PetscPrintf(PETSC_COMM_WORLD, "read 3D mesh ...\n");
        
        std::ifstream file("OCFD-grid.dat", std::ios::binary | std::ios::in);
        if (!file.is_open()) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
                    "Cannot open OCFD-grid.dat");
        }
        
        // 读取所有网格点的 x, y, z 坐标
        file.read(reinterpret_cast<char*>(x3), 
                  nx_global * ny_global * nz_global * sizeof(PetscReal));
        file.read(reinterpret_cast<char*>(y3), 
                  nx_global * ny_global * nz_global * sizeof(PetscReal));
        file.read(reinterpret_cast<char*>(z3), 
                  nx_global * ny_global * nz_global * sizeof(PetscReal));
        
        file.close();
    }
    
    // 广播到所有进程
    MPI_Bcast(x3, nx_global * ny_global * nz_global, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    MPI_Bcast(y3, nx_global * ny_global * nz_global, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    MPI_Bcast(z3, nx_global * ny_global * nz_global, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    
    ierr = DMDACreate3d(PETSC_COMM_WORLD,
                        DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                        DMDA_STENCIL_BOX,
                        nx_global, ny_global, nz_global,
                        npx0, npy0, npz0,
                        1, LAP,
                        NULL, NULL, NULL,
                        &da);
    CHKERRQ(ierr);
    
    ierr = DMSetUp(da);
    CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(da, &Axx);
    CHKERRQ(ierr);
    ierr = VecDuplicate(Axx, &Ayy);
    CHKERRQ(ierr);
    ierr = VecDuplicate(Axx, &Azz);
    CHKERRQ(ierr);
    
    DMDALocalInfo info;
    ierr = DMDAGetLocalInfo(da, &info);
    CHKERRQ(ierr);
    
    nx = info.xm; ny = info.ym; nz = info.zm;
    
    // 获取局部数组并填充网格坐标
    PetscReal ***axx, ***ayy, ***azz;
    ierr = DMDAVecGetArray(da, Axx, &axx);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Ayy, &ayy);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Azz, &azz);
    CHKERRQ(ierr);
    
    // 使用全局索引填充当前进程的局部网格
    for (PetscInt k = info.zs; k < info.zs + info.zm; k++) {
        for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
            for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
                // 3D 网格的线性索引
                PetscInt idx = k * nx_global * ny_global + j * nx_global + i;
                axx[k][j][i] = x3[idx];
                ayy[k][j][i] = y3[idx];
                azz[k][j][i] = z3[idx];
            }
        }
    }
    
    ierr = DMDAVecRestoreArray(da, Axx, &axx);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Ayy, &ayy);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Azz, &azz);
    CHKERRQ(ierr);
    
    // 释放临时数组
    delete[] x3;
    delete[] y3;
    delete[] z3;
    
    // 如果是预计算的雅可比类型，则创建雅可比向量（但不计算）
    if (Iflag_Gridtype == GRID_AND_JACOBIAN3D) {
        ierr = DMCreateGlobalVector(da, &Akx);
        CHKERRQ(ierr);
        ierr = VecDuplicate(Akx, &Aky); CHKERRQ(ierr);
        ierr = VecDuplicate(Akx, &Akz); CHKERRQ(ierr);
        ierr = VecDuplicate(Akx, &Aix); CHKERRQ(ierr);
        ierr = VecDuplicate(Akx, &Aiy); CHKERRQ(ierr);
        ierr = VecDuplicate(Akx, &Aiz); CHKERRQ(ierr);
        ierr = VecDuplicate(Akx, &Asx); CHKERRQ(ierr);
        ierr = VecDuplicate(Akx, &Asy); CHKERRQ(ierr);
        ierr = VecDuplicate(Akx, &Asz); CHKERRQ(ierr);
        ierr = VecDuplicate(Akx, &Ajac); CHKERRQ(ierr);
    }
    
    if (rank == 0) {
        PetscPrintf(PETSC_COMM_WORLD, "read 3D mesh OK\n");
    }
    
    return 0;
}

PetscErrorCode Mesh::comput_Jacobian3d()
{
    PetscErrorCode ierr;
    
    // 创建雅可比系数向量
    ierr = DMCreateGlobalVector(da, &Akx);
    CHKERRQ(ierr);
    ierr = VecDuplicate(Akx, &Aky); CHKERRQ(ierr);
    ierr = VecDuplicate(Akx, &Akz); CHKERRQ(ierr);
    ierr = VecDuplicate(Akx, &Aix); CHKERRQ(ierr);
    ierr = VecDuplicate(Akx, &Aiy); CHKERRQ(ierr);
    ierr = VecDuplicate(Akx, &Aiz); CHKERRQ(ierr);
    ierr = VecDuplicate(Akx, &Asx); CHKERRQ(ierr);
    ierr = VecDuplicate(Akx, &Asy); CHKERRQ(ierr);
    ierr = VecDuplicate(Akx, &Asz); CHKERRQ(ierr);
    ierr = VecDuplicate(Akx, &Ajac); CHKERRQ(ierr);
    ierr = VecDuplicate(Akx, &Akx1); CHKERRQ(ierr);
    ierr = VecDuplicate(Akx, &Aky1); CHKERRQ(ierr);
    ierr = VecDuplicate(Akx, &Akz1); CHKERRQ(ierr);
    ierr = VecDuplicate(Akx, &Aix1); CHKERRQ(ierr);
    ierr = VecDuplicate(Akx, &Aiy1); CHKERRQ(ierr);
    ierr = VecDuplicate(Akx, &Aiz1); CHKERRQ(ierr);
    ierr = VecDuplicate(Akx, &Asx1); CHKERRQ(ierr);
    ierr = VecDuplicate(Akx, &Asy1); CHKERRQ(ierr);
    ierr = VecDuplicate(Akx, &Asz1); CHKERRQ(ierr);
    

    // 获取局部数组
    DMDALocalInfo info;
    ierr = DMDAGetLocalInfo(da, &info);
    CHKERRQ(ierr);

    PetscReal ***axx, ***ayy, ***azz;
    PetscReal ***akx, ***aky, ***akz;
    PetscReal ***aix, ***aiy, ***aiz;
    PetscReal ***asx, ***asy, ***asz;
    PetscReal ***akx1, ***aky1, ***akz1;
    PetscReal ***aix1, ***aiy1, ***aiz1;
    PetscReal ***asx1, ***asy1, ***asz1;
    PetscReal ***ajac;
    
    ierr = DMDAVecGetArray(da, Axx, &axx);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Ayy, &ayy);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Azz, &azz);
    CHKERRQ(ierr);
    
    ierr = DMDAVecGetArray(da, Akx, &akx);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Aky, &aky);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Akz, &akz);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Aix, &aix);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Aiy, &aiy);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Aiz, &aiz);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Asx, &asx);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Asy, &asy);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Asz, &asz);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Akx1, &akx1);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Aky1, &aky1);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Akz1, &akz1);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Aix1, &aix1);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Aiy1, &aiy1);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Aiz1, &aiz1);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Asx1, &asx1);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Asy1, &asy1);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Asz1, &asz1);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Ajac, &ajac);
    CHKERRQ(ierr);
    
    // 分配临时数组存储导数（与 Fortran 版本一致）
    PetscReal ***xi, ***xj, ***xk;
    PetscReal ***yi, ***yj, ***yk;
    PetscReal ***zi, ***zj, ***zk;
    
    // 获取局部网格范围（不含 ghost points）
    PetscInt xs = info.xs, ys = info.ys, zs = info.zs;
    PetscInt xm = info.xm, ym = info.ym, zm = info.zm;
    
    // 分配三维数组
    ierr = PetscMalloc(zm * sizeof(PetscReal**), &xi);
    CHKERRQ(ierr);
    ierr = PetscMalloc(zm * sizeof(PetscReal**), &xj);
    CHKERRQ(ierr);
    ierr = PetscMalloc(zm * sizeof(PetscReal**), &xk);
    CHKERRQ(ierr);
    ierr = PetscMalloc(zm * sizeof(PetscReal**), &yi);
    CHKERRQ(ierr);
    ierr = PetscMalloc(zm * sizeof(PetscReal**), &yj);
    CHKERRQ(ierr);
    ierr = PetscMalloc(zm * sizeof(PetscReal**), &yk);
    CHKERRQ(ierr);
    ierr = PetscMalloc(zm * sizeof(PetscReal**), &zi);
    CHKERRQ(ierr);
    ierr = PetscMalloc(zm * sizeof(PetscReal**), &zj);
    CHKERRQ(ierr);
    ierr = PetscMalloc(zm * sizeof(PetscReal**), &zk);
    CHKERRQ(ierr);
    
    for (PetscInt k = 0; k < zm; k++) {
        ierr = PetscMalloc(ym * sizeof(PetscReal*), &xi[k]);
        CHKERRQ(ierr);
        ierr = PetscMalloc(ym * sizeof(PetscReal*), &xj[k]);
        CHKERRQ(ierr);
        ierr = PetscMalloc(ym * sizeof(PetscReal*), &xk[k]);
        CHKERRQ(ierr);
        ierr = PetscMalloc(ym * sizeof(PetscReal*), &yi[k]);
        CHKERRQ(ierr);
        ierr = PetscMalloc(ym * sizeof(PetscReal*), &yj[k]);
        CHKERRQ(ierr);
        ierr = PetscMalloc(ym * sizeof(PetscReal*), &yk[k]);
        CHKERRQ(ierr);
        ierr = PetscMalloc(ym * sizeof(PetscReal*), &zi[k]);
        CHKERRQ(ierr);
        ierr = PetscMalloc(ym * sizeof(PetscReal*), &zj[k]);
        CHKERRQ(ierr);
        ierr = PetscMalloc(ym * sizeof(PetscReal*), &zk[k]);
        CHKERRQ(ierr);
        
        for (PetscInt j = 0; j < ym; j++) {
            ierr = PetscMalloc(xm * sizeof(PetscReal), &xi[k][j]);
            CHKERRQ(ierr);
            ierr = PetscMalloc(xm * sizeof(PetscReal), &xj[k][j]);
            CHKERRQ(ierr);
            ierr = PetscMalloc(xm * sizeof(PetscReal), &xk[k][j]);
            CHKERRQ(ierr);
            ierr = PetscMalloc(xm * sizeof(PetscReal), &yi[k][j]);
            CHKERRQ(ierr);
            ierr = PetscMalloc(xm * sizeof(PetscReal), &yj[k][j]);
            CHKERRQ(ierr);
            ierr = PetscMalloc(xm * sizeof(PetscReal), &yk[k][j]);
            CHKERRQ(ierr);
            ierr = PetscMalloc(xm * sizeof(PetscReal), &zi[k][j]);
            CHKERRQ(ierr);
            ierr = PetscMalloc(xm * sizeof(PetscReal), &zj[k][j]);
            CHKERRQ(ierr);
            ierr = PetscMalloc(xm * sizeof(PetscReal), &zk[k][j]);
            CHKERRQ(ierr);
        }
    }
    
    // 计算各方向导数（根据Scheme_Vis选择CD6或CD8）
    compute_derivative_x(axx, xi);
    compute_derivative_x(ayy, yi);
    compute_derivative_x(azz, zi);
    
    compute_derivative_y(axx, xj);
    compute_derivative_y(ayy, yj);
    compute_derivative_y(azz, zj);
    
    compute_derivative_z(axx, xk);
    compute_derivative_z(ayy, yk);
    compute_derivative_z(azz, zk);
    
    // 计算雅可比系数（与 Fortran 代码完全一致）
    for (PetscInt k = zs; k < zs + zm; k++) {
        for (PetscInt j = ys; j < ys + ym; j++) {
            for (PetscInt i = xs; i < xs + xm; i++) {
                PetscReal xi1 = xi[k][j][i];
                PetscReal xj1 = xj[k][j][i];
                PetscReal xk1 = xk[k][j][i];
                PetscReal yi1 = yi[k][j][i];
                PetscReal yj1 = yj[k][j][i];
                PetscReal yk1 = yk[k][j][i];
                PetscReal zi1 = zi[k][j][i];
                PetscReal zj1 = zj[k][j][i];
                PetscReal zk1 = zk[k][j][i];
                
                // 计算雅可比行列式（与 Fortran 完全一致）
                // Jac1 = 1/J = 1/det(d(x,y,z)/d(ξ,η,ζ))
                PetscReal Jac1 = 1.0 / (xi1*yj1*zk1 + yi1*zj1*xk1 + zi1*xj1*yk1 
                                       - zi1*yj1*xk1 - yi1*xj1*zk1 - xi1*zj1*yk1);
                
                ajac[k][j][i] = Jac1;
                
                // 计算雅可比系数（与 Fortran 完全一致）
                akx[k][j][i] = Jac1 * (yj1*zk1 - zj1*yk1);
                aky[k][j][i] = Jac1 * (zj1*xk1 - xj1*zk1);
                akz[k][j][i] = Jac1 * (xj1*yk1 - yj1*xk1);
                aix[k][j][i] = Jac1 * (yk1*zi1 - zk1*yi1);
                aiy[k][j][i] = Jac1 * (zk1*xi1 - xk1*zi1);
                aiz[k][j][i] = Jac1 * (xk1*yi1 - yk1*xi1);
                asx[k][j][i] = Jac1 * (yi1*zj1 - zi1*yj1);
                asy[k][j][i] = Jac1 * (zi1*xj1 - xi1*zj1);
                asz[k][j][i] = Jac1 * (xi1*yj1 - yi1*xj1);
                
                akx1[k][j][i] = akx[k][j][i] / ajac[k][j][i];
                aky1[k][j][i] = aky[k][j][i] / ajac[k][j][i];
                akz1[k][j][i] = akz[k][j][i] / ajac[k][j][i];
                aix1[k][j][i] = aix[k][j][i] / ajac[k][j][i];
                aiy1[k][j][i] = aiy[k][j][i] / ajac[k][j][i];
                aiz1[k][j][i] = aiz[k][j][i] / ajac[k][j][i];
                asx1[k][j][i] = asx[k][j][i] / ajac[k][j][i];
                asy1[k][j][i] = asy[k][j][i] / ajac[k][j][i];
                asz1[k][j][i] = asz[k][j][i] / ajac[k][j][i];
                
                // 检查雅可比是否为负（与 Fortran 一致）
                if (Jac1 < 0) {
                    PetscPrintf(PETSC_COMM_WORLD, 
                               " Jocabian < 0 !!! , Jac=%f\n", Jac1);
                }
            }
        }
    }
    
    // 释放临时数组
    for (PetscInt k = 0; k < zm; k++) {
        for (PetscInt j = 0; j < ym; j++) {
            PetscFree(xi[k][j]);
            PetscFree(xj[k][j]);
            PetscFree(xk[k][j]);
            PetscFree(yi[k][j]);
            PetscFree(yj[k][j]);
            PetscFree(yk[k][j]);
            PetscFree(zi[k][j]);
            PetscFree(zj[k][j]);
            PetscFree(zk[k][j]);
        }
        PetscFree(xi[k]);
        PetscFree(xj[k]);
        PetscFree(xk[k]);
        PetscFree(yi[k]);
        PetscFree(yj[k]);
        PetscFree(yk[k]);
        PetscFree(zi[k]);
        PetscFree(zj[k]);
        PetscFree(zk[k]);
    }
    PetscFree(xi);
    PetscFree(xj);
    PetscFree(xk);
    PetscFree(yi);
    PetscFree(yj);
    PetscFree(yk);
    PetscFree(zi);
    PetscFree(zj);
    PetscFree(zk);
    
    return 0;
}

// MPI边界交换接口（留空给您实现）
void Mesh::exchange_boundary_xyz(Vec &f)
{
    // TODO: 您需要实现MPI边界交换
    // 这里可以使用PETSc的VecGhostUpdate或自定义MPI通信
}

// Ghost Cell边界处理接口（留空给您实现）
PetscErrorCode Mesh::Jac_Ghost_boundary()
{
    PetscErrorCode ierr = 0;
    // TODO: 您需要实现Ghost Cell边界处理
    return ierr;
}

// 导出Tecplot格式文件
#if 1
PetscErrorCode Mesh::ExportToTecplot(const std::string &filename)
{
    PetscErrorCode ierr;
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    
    DMDALocalInfo info;
    PetscReal ***axx, ***ayy, ***azz;
    PetscReal ***akx, ***aky, ***akz;
    PetscReal ***aix, ***aiy, ***aiz;
    PetscReal ***asx, ***asy, ***asz;
    PetscReal ***ajac;
    PetscReal ***akx1, ***aky1, ***akz1;
    PetscReal ***aix1, ***aiy1, ***aiz1;
    PetscReal ***asx1, ***asy1, ***asz1;
    
    ierr = GetLocalArrays(axx, ayy, azz, akx, aky, akz, 
                          aix, aiy, aiz, asx, asy, asz, ajac,
                          akx1, aky1, akz1, aix1, aiy1, aiz1, 
                          asx1, asy1, asz1, info);
    CHKERRQ(ierr);
    
    // 每个进程写入自己的文件
    std::string proc_filename = filename + "_proc" + std::to_string(rank) + ".dat";
    std::ofstream outfile(proc_filename);
    
    if (!outfile.is_open()) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Cannot open output file");
    }
    
    // 写入Tecplot文件头
    outfile << "TITLE = \"Mesh and Jacobian Data\"" << std::endl;
    outfile << "VARIABLES = \"X\", \"Y\", \"Z\", \"Axx\", \"Akx\", \"Akx1\"" << std::endl;
    outfile << "ZONE T=\"Process " << rank << "\", I=" << info.xm 
            << ", J=" << info.ym << ", K=" << info.zm << ", F=POINT" << std::endl;
    
    // 写入数据点
    for (PetscInt k = info.zs; k < info.zs + info.zm; k++) {
        for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
            for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
                outfile << axx[k][j][i] << " "
                        << ayy[k][j][i] << " "
                        << azz[k][j][i] << " "
                        << axx[k][j][i] << " "
                        << akx[k][j][i] << " "
                        << akx1[k][j][i] << std::endl;
            }
        }
    }
    
    outfile.close();
    
    ierr = RestoreLocalArrays(axx, ayy, azz, akx, aky, akz, 
                              aix, aiy, aiz, asx, asy, asz, ajac,
                              akx1, aky1, akz1, aix1, aiy1, aiz1, 
                              asx1, asy1, asz1);
    CHKERRQ(ierr);
    
    if (rank == 0) {
        PetscPrintf(PETSC_COMM_WORLD, "Data exported to %s*\n", filename.c_str());
    }
    
    return 0;
}
#endif

#if 0
PetscErrorCode Mesh::ExportToTecplot(const std::string &filename)
{
    PetscErrorCode ierr;
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    
    DMDALocalInfo info;
    ierr = DMDAGetLocalInfo(da, &info);
    CHKERRQ(ierr);
    
    PetscReal ***axx, ***ayy, ***azz;
    
    ierr = DMDAVecGetArray(da, Axx, &axx);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Ayy, &ayy);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Azz, &azz);
    CHKERRQ(ierr);
    
    std::string proc_filename = filename + "_proc" + std::to_string(rank) + ".dat";
    std::ofstream outfile(proc_filename);
    
    if (!outfile.is_open()) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Cannot open output file");
    }
    
    outfile << "TITLE = \"Mesh Data\"" << std::endl;
    outfile << "VARIABLES = \"X\", \"Y\", \"Z\"" << std::endl;
    outfile << "ZONE T=\"Process " << rank << "\", I=" << info.xm 
            << ", J=" << info.ym << ", K=" << info.zm << ", F=POINT" << std::endl;
    
    for (PetscInt k = info.zs; k < info.zs + info.zm; k++) {
        for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
            for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
                outfile << axx[k][j][i] << " "
                        << ayy[k][j][i] << " "
                        << azz[k][j][i] << std::endl;
            }
        }
    }
    
    outfile.close();
    
    ierr = DMDAVecRestoreArray(da, Axx, &axx);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Ayy, &ayy);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Azz, &azz);
    CHKERRQ(ierr);
    
    if (rank == 0) {
        PetscPrintf(PETSC_COMM_WORLD, "Data exported to %s*\n", filename.c_str());
    }
    
    return 0;
}
#endif

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

// 打印网格信息
void Mesh::printInfo() const
{
    PetscPrintf(PETSC_COMM_WORLD,
                "Mesh Info:\n"
                "  Global size: %ld x %ld x %ld\n"
                "  Local size:  %ld x %ld x %ld\n"
                "  Grid type:   %d\n"
                "  My ID:       %d, Process division: (%d,%d,%d)\n",
                (long)nx_global, (long)ny_global, (long)nz_global,
                (long)nx, (long)ny, (long)nz,
                (int)Iflag_Gridtype,
                (int)my_id, (int)npx0, (int)npy0, (int)npz0);
}